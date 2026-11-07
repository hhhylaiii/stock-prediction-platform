import copy
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pytest
import responses
from django.core.management import call_command

from apps.etl.adapters.twse import TwseAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import EtlRun, RawTwse
from apps.etl.ratelimit import RateLimiter, RetryableError
from apps.market.calendar import NotATradingDayError, TradingDays
from apps.market.models import DispositionEvent

BASE = "https://www.twse.com.tw"
NOTICE = f"{BASE}/rwd/zh/announcement/notice"
PUNISH = f"{BASE}/rwd/zh/announcement/punish"
TWT85U = f"{BASE}/exchangeReport/TWT85U"
DAY = date(2026, 10, 28)
NO_DATA = {"stat": "很抱歉，沒有符合條件的資料!"}


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TwseAdapter:
    return TwseAdapter(october_2026, twse_limiter)


@pytest.fixture
def adapter_2010(january_2010: TradingDays, twse_limiter: RateLimiter) -> TwseAdapter:
    return TwseAdapter(january_2010, twse_limiter)


def _events(adapter: TwseAdapter, dataset: str, start: date, end: date) -> set[tuple[Any, ...]]:
    frame = adapter.normalize(dataset, adapter.fetch(dataset, start, end))
    return {tuple(row) for row in frame[["stock_id", "start_date", "type", "end_date"]].values}


def _sent(call_index: int) -> dict[str, str]:
    url = responses.calls[call_index].request.url or ""
    return dict(parse_qsl(urlsplit(url).query))


@responses.activate
@pytest.mark.django_db
def test_attention_looks_back_one_session_and_keeps_one_day_events(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("twse/disposition/notice_20261028.json")
    responses.get(NOTICE, json=payload)
    events = _events(adapter, "attention", DAY, DAY)
    assert _sent(0) == {
        "querytype": "1",
        "startDate": "20261027",
        "endDate": "20261028",
        "response": "json",
    }
    assert events == {
        (code, DAY, "attention", DAY) for code in ("050503", "1303", "2897B", "9103", "911608")
    }
    assert RawTwse.objects.get().payload == payload


@responses.activate
@pytest.mark.django_db
def test_attention_of_2010_with_dotted_roc_dates(
    adapter_2010: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(NOTICE, json=load_fixture("twse/disposition/notice_20100104_05.json"))
    events = _events(adapter_2010, "attention", date(2010, 1, 5), date(2010, 1, 5))
    assert _sent(0)["startDate"] == "20100104"
    assert ("2496", date(2010, 1, 4), "attention", date(2010, 1, 4)) in events
    assert ("2496", date(2010, 1, 5), "attention", date(2010, 1, 5)) in events
    assert len(events) == 10


@responses.activate
@pytest.mark.django_db
def test_attention_listing_a_stock_twice_on_a_day_keeps_one_event(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("twse/disposition/notice_20261028.json")
    payload["data"].append(copy.deepcopy(payload["data"][2]))
    payload["total"] = len(payload["data"])
    responses.get(NOTICE, json=payload)
    adapter.upsert(
        "attention", adapter.normalize("attention", adapter.fetch("attention", DAY, DAY))
    )
    assert DispositionEvent.objects.filter(stock_id="1303").count() == 1


@responses.activate
@pytest.mark.django_db
def test_attention_at_the_start_of_the_calendar_queries_that_day(adapter: TwseAdapter) -> None:
    responses.get(NOTICE, json=NO_DATA)
    first = date(2026, 10, 1)
    assert _events(adapter, "attention", first, first) == set()
    assert (_sent(0)["startDate"], _sent(0)["endDate"]) == ("20261001", "20261001")
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_disposition_month_query_keeps_periods_overlapping_it(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(PUNISH, json=load_fixture("twse/disposition/punish_202610.json"))
    events = _events(adapter, "disposition", date(2026, 10, 1), date(2026, 10, 30))
    assert _sent(0) == {"startDate": "20261001", "endDate": "20261030", "response": "json"}
    assert events == {
        ("053040", date(2026, 10, 20), "disposition", date(2026, 10, 27)),
        ("053040", date(2026, 10, 15), "disposition", date(2026, 10, 21)),
        ("071804", date(2026, 10, 28), "disposition", date(2026, 11, 3)),
        ("071804", date(2026, 10, 13), "disposition", date(2026, 10, 19)),
        ("2466", date(2026, 10, 30), "disposition", date(2026, 11, 5)),
        ("2466", date(2026, 10, 14), "disposition", date(2026, 10, 20)),
        ("2466", date(2026, 10, 8), "disposition", date(2026, 10, 15)),
        ("6226", date(2026, 9, 21), "disposition", date(2026, 10, 1)),
        ("911608", date(2026, 10, 19), "disposition", date(2026, 10, 23)),
    }


@responses.activate
@pytest.mark.django_db
def test_disposition_of_a_day_queries_that_day_only(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("twse/disposition/punish_202610.json")
    payload["data"] = [row for row in payload["data"] if row[6] == "115/10/28～115/11/03"]
    payload["total"] = len(payload["data"])
    responses.get(PUNISH, json=payload)
    events = _events(adapter, "disposition", DAY, DAY)
    assert (_sent(0)["startDate"], _sent(0)["endDate"]) == ("20261028", "20261028")
    assert events == {("071804", DAY, "disposition", date(2026, 11, 3))}


@responses.activate
@pytest.mark.django_db
def test_disposition_announced_in_the_query_but_starting_later_is_kept(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("twse/disposition/punish_202610.json")
    payload["data"] = [row for row in payload["data"] if row[1] == "115/10/29"]
    payload["total"] = len(payload["data"])
    responses.get(PUNISH, json=payload)
    day = date(2026, 10, 29)
    assert _events(adapter, "disposition", day, day) == {
        ("2466", date(2026, 10, 30), "disposition", date(2026, 11, 5))
    }


@responses.activate
@pytest.mark.django_db
def test_disposition_announced_twice_keeps_the_later_announcement(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("twse/disposition/punish_202610.json")
    first = next(row for row in payload["data"] if row[6] == "115/10/20～115/10/27")
    later = [*first[:1], "115/10/26", *first[2:6], "115/10/20～115/10/29", *first[7:]]
    payload["data"] = [later, first]
    payload["total"] = 2
    responses.get(PUNISH, json=payload)
    adapter.upsert(
        "disposition",
        adapter.normalize(
            "disposition", adapter.fetch("disposition", date(2026, 10, 1), date(2026, 10, 30))
        ),
    )
    assert DispositionEvent.objects.get().end_date == date(2026, 10, 29)


@responses.activate
@pytest.mark.django_db
def test_attention_on_a_non_trading_day_is_a_schema_change(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("twse/disposition/notice_20261028.json")
    payload["data"][0][5] = "115.10.24"
    responses.get(NOTICE, json=payload)
    with pytest.raises(SchemaChangedError, match="2026-10-24"):
        adapter.fetch("attention", date(2026, 10, 27), date(2026, 10, 28))


@responses.activate
@pytest.mark.django_db
def test_disposition_period_outside_the_query_is_a_schema_change(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(PUNISH, json=load_fixture("twse/disposition/punish_202610.json"))
    with pytest.raises(SchemaChangedError, match="053040"):
        adapter.fetch("disposition", DAY, DAY)
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_disposition_of_2010_across_the_month_end(
    adapter_2010: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(PUNISH, json=load_fixture("twse/disposition/punish_201001.json"))
    events = _events(adapter_2010, "disposition", date(2010, 1, 1), date(2010, 1, 31))
    assert _sent(0)["startDate"] == "20100104"
    assert ("1439", date(2010, 1, 27), "disposition", date(2010, 2, 2)) in events
    assert len(events) == 7


@responses.activate
@pytest.mark.django_db
def test_rerun_moves_the_end_of_a_postponed_disposition(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("twse/disposition/punish_202610.json")
    october = (date(2026, 10, 1), date(2026, 10, 30))
    responses.get(PUNISH, json=payload)
    adapter.upsert(
        "disposition", adapter.normalize("disposition", adapter.fetch("disposition", *october))
    )
    later = copy.deepcopy(payload)
    later["data"][0][6] = "115/10/20～115/10/28"
    responses.get(PUNISH, json=later)
    adapter.upsert(
        "disposition", adapter.normalize("disposition", adapter.fetch("disposition", *october))
    )
    assert DispositionEvent.objects.count() == 9
    moved = DispositionEvent.objects.get(stock_id="053040", start_date=date(2026, 10, 20))
    assert moved.end_date == date(2026, 10, 28)


@responses.activate
@pytest.mark.django_db
def test_full_delivery_lists_each_stock_for_the_day(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(TWT85U, json=load_fixture("twse/disposition/twt85u_20261028.json"))
    events = _events(adapter, "full_delivery", DAY, DAY)
    assert _sent(0) == {"date": "20261028", "response": "json"}
    assert len(events) == 10
    assert ("2314", DAY, "full_delivery", DAY) in events


@responses.activate
@pytest.mark.django_db
def test_full_delivery_of_2010(
    adapter_2010: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(TWT85U, json=load_fixture("twse/disposition/twt85u_20100104.json"))
    day = date(2010, 1, 4)
    assert _events(adapter_2010, "full_delivery", day, day) == {
        ("1432", day, "full_delivery", day),
        ("1438", day, "full_delivery", day),
    }


@responses.activate
@pytest.mark.django_db
def test_full_delivery_list_with_nothing_listed_is_not_ready(adapter: TwseAdapter) -> None:
    responses.get(TWT85U, json=NO_DATA)
    with pytest.raises(SourceNotReadyError, match="TWT85U"):
        adapter.fetch("full_delivery", DAY, DAY)
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_foreign_currency_lines_are_left_out(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("twse/disposition/notice_20261028.json")
    payload["data"][0][1] = "00636K"
    responses.get(NOTICE, json=payload)
    events = _events(adapter, "attention", DAY, DAY)
    assert {event[0] for event in events} == {"1303", "2897B", "9103", "911608"}


@responses.activate
@pytest.mark.django_db
def test_full_delivery_of_the_previous_day_is_not_ready(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("twse/disposition/twt85u_20261028.json")
    responses.get(TWT85U, json={**payload, "date": "20261027"})
    with pytest.raises(SourceNotReadyError, match="still serves 20261027"):
        adapter.fetch("full_delivery", DAY, DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("url", "dataset", "fixture", "change", "message"),
    [
        (
            PUNISH,
            "disposition",
            "punish_202610.json",
            lambda p: p["fields"].__setitem__(6, "處置期間"),
            "處置起迄時間",
        ),
        (
            PUNISH,
            "disposition",
            "punish_202610.json",
            lambda p: p["data"][0].__setitem__(6, "115/10/20"),
            "period",
        ),
        (PUNISH, "disposition", "punish_202610.json", lambda p: p["data"].pop(), "rows"),
        (
            NOTICE,
            "attention",
            "notice_20261028.json",
            lambda p: p["data"][0].__setitem__(5, "1028"),
            "ROC",
        ),
        (
            NOTICE,
            "attention",
            "notice_20261028.json",
            lambda p: p["data"][0].__setitem__(1, "05 03"),
            "code",
        ),
        (
            TWT85U,
            "full_delivery",
            "twt85u_20261028.json",
            lambda p: p["fields"].__setitem__(0, "代號"),
            "證券代號",
        ),
        (
            PUNISH,
            "disposition",
            "punish_202610.json",
            lambda p: p["data"][0].__setitem__(6, "115/10/27～115/10/20"),
            "ends before",
        ),
        (PUNISH, "disposition", "punish_202610.json", lambda p: p["data"][0].pop(), "do not match"),
        (
            PUNISH,
            "disposition",
            "punish_202610.json",
            lambda p: p.__setitem__("stat", "ERROR"),
            "stat",
        ),
    ],
    ids=[
        "period-renamed",
        "period-text",
        "truncated",
        "date-text",
        "bad-code",
        "code-renamed",
        "period-reversed",
        "short-row",
        "stat",
    ],
)
def test_layout_change_is_a_schema_change_and_keeps_raw(
    load_fixture: Callable[[str], Any],
    october_2026: TradingDays,
    twse_limiter: RateLimiter,
    url: str,
    dataset: str,
    fixture: str,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    adapter = TwseAdapter(october_2026, twse_limiter)
    payload = load_fixture(f"twse/disposition/{fixture}")
    change(payload)
    responses.get(url, json=payload)
    start = date(2026, 10, 1) if dataset == "disposition" else DAY
    end = date(2026, 10, 30) if dataset == "disposition" else DAY
    with pytest.raises(SchemaChangedError, match=message):
        adapter.fetch(dataset, start, end)
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(("url", "dataset"), [(NOTICE, "attention"), (PUNISH, "disposition")])
def test_empty_list_writes_nothing_and_keeps_the_answer_raw(
    adapter: TwseAdapter, url: str, dataset: str
) -> None:
    responses.get(url, json=NO_DATA)
    frame = adapter.normalize(dataset, adapter.fetch(dataset, DAY, DAY))
    assert adapter.upsert(dataset, frame) == 0
    assert RawTwse.objects.get().payload == NO_DATA


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(("url", "dataset"), [(NOTICE, "attention"), (PUNISH, "disposition")])
def test_a_busy_answer_is_retried_later_and_not_kept(
    adapter: TwseAdapter, url: str, dataset: str
) -> None:
    responses.get(url, json={"stat": "系統忙碌中，請稍後再試！"})
    with pytest.raises(RetryableError, match="busy"):
        adapter.fetch(dataset, DAY, DAY)
    assert not RawTwse.objects.exists()


@responses.activate
@pytest.mark.django_db
def test_a_busy_answer_is_retried_in_place_and_the_next_answer_kept(
    adapter: TwseAdapter,
) -> None:
    responses.get(PUNISH, json={"stat": "系統忙碌中，請稍後再試！"})
    responses.get(PUNISH, json=NO_DATA)
    frame = adapter.normalize("disposition", adapter.fetch("disposition", DAY, DAY))
    assert frame.empty
    assert len(responses.calls) == 2
    assert RawTwse.objects.get().payload == NO_DATA


@responses.activate
@pytest.mark.django_db
def test_a_lowercase_ok_is_an_answer(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("twse/disposition/notice_20261028.json")
    responses.get(NOTICE, json={**payload, "stat": "ok"})
    assert len(_events(adapter, "attention", DAY, DAY)) == 5


@responses.activate
@pytest.mark.parametrize("dataset", ["attention", "disposition", "full_delivery"])
def test_non_trading_day_is_rejected_before_any_request(adapter: TwseAdapter, dataset: str) -> None:
    with pytest.raises(NotATradingDayError):
        adapter.fetch(dataset, date(2026, 10, 25), date(2026, 10, 25))
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_fetch_command_backfills_a_month_of_dispositions_in_one_request(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    responses.get(PUNISH, json=load_fixture("twse/disposition/punish_202610.json"))
    for _ in range(2):
        call_command(
            "fetch", source="twse", dataset="disposition", start="2026-10-01", end="2026-10-30"
        )
    assert len(responses.calls) == 2
    assert DispositionEvent.objects.count() == 9
    run = EtlRun.objects.filter(dataset="disposition").latest("run_id")
    assert (run.status, run.rows, run.checkpoint) == ("success", 9, {"last_date": "2026-10-30"})


def _with_end_of_september(october_2026: TradingDays) -> TradingDays:
    october = october_2026.between(date(2026, 10, 1), date(2026, 10, 30))
    september = (date(2026, 9, 29), date(2026, 9, 30))
    return TradingDays((*september, *october), date(2026, 9, 29), date(2026, 10, 31))


def _september_rows(payload: dict[str, Any], rows: list[list[Any]]) -> dict[str, Any]:
    return {**payload, "data": rows, "total": len(rows)}


@responses.activate
@pytest.mark.django_db
def test_backfill_across_months_sends_one_request_per_month(
    october_2026: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    adapter = TwseAdapter(_with_end_of_september(october_2026), twse_limiter)
    october = load_fixture("twse/disposition/punish_202610.json")
    september = _september_rows(october, [row for row in october["data"] if row[2] == "6226"])
    responses.get(PUNISH, json=september)
    responses.get(PUNISH, json=october)
    run = adapter.run("disposition", date(2026, 9, 29), date(2026, 10, 30))
    assert [(_sent(n)["startDate"], _sent(n)["endDate"]) for n in range(2)] == [
        ("20260929", "20260930"),
        ("20261001", "20261030"),
    ]
    assert (run.rows, run.checkpoint) == (10, {"last_date": "2026-10-30"})
    assert DispositionEvent.objects.count() == 9


@responses.activate
@pytest.mark.django_db
def test_attention_backfill_looks_back_across_the_month_boundary(
    october_2026: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    adapter = TwseAdapter(_with_end_of_september(october_2026), twse_limiter)
    responses.get(NOTICE, json=NO_DATA)
    responses.get(NOTICE, json=load_fixture("twse/disposition/notice_20261028.json"))
    run = adapter.run("attention", date(2026, 9, 29), date(2026, 10, 30))
    assert [(_sent(n)["startDate"], _sent(n)["endDate"]) for n in range(2)] == [
        ("20260929", "20260930"),
        ("20260930", "20261030"),
    ]
    assert run.rows == 5


@responses.activate
@pytest.mark.django_db
def test_failed_month_keeps_the_months_before_it(
    october_2026: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    adapter = TwseAdapter(_with_end_of_september(october_2026), twse_limiter)
    october = load_fixture("twse/disposition/punish_202610.json")
    responses.get(
        PUNISH, json=_september_rows(october, [row for row in october["data"] if row[2] == "6226"])
    )
    responses.get(PUNISH, json={**october, "total": 99})
    with pytest.raises(SchemaChangedError, match="of 99 rows"):
        adapter.run("disposition", date(2026, 9, 29), date(2026, 10, 30))
    run = EtlRun.objects.get()
    assert (run.status, run.rows, run.checkpoint) == ("failed", 1, {"last_date": "2026-09-30"})
    assert DispositionEvent.objects.get().stock_id == "6226"
