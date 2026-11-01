import copy
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.parse import parse_qs

import pytest
import responses
from django.core.management import call_command

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import EtlRun, RawTpex
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import NotATradingDayError, TradingDays
from apps.market.models import DispositionEvent

BASE = "https://www.tpex.org.tw/www/zh-tw"
ATTENTION = f"{BASE}/bulletin/attention"
DISPOSAL = f"{BASE}/bulletin/disposal"
CHTM = f"{BASE}/afterTrading/chtm"
DAY = date(2026, 10, 28)
OCTOBER = (date(2026, 10, 1), date(2026, 10, 30))


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TpexAdapter:
    return TpexAdapter(october_2026, twse_limiter)


@pytest.fixture
def adapter_2010(january_2010: TradingDays, twse_limiter: RateLimiter) -> TpexAdapter:
    return TpexAdapter(january_2010, twse_limiter)


def _sent(call_index: int) -> dict[str, str]:
    body = responses.calls[call_index].request.body or ""
    text = body.decode() if isinstance(body, bytes) else body
    return {key: values[0] for key, values in parse_qs(text, keep_blank_values=True).items()}


def _events(adapter: TpexAdapter, dataset: str, start: date, end: date) -> set[tuple[Any, ...]]:
    frame = adapter.normalize(dataset, adapter.fetch(dataset, start, end))
    return {tuple(row) for row in frame[["stock_id", "start_date", "type", "end_date"]].values}


def _within(payload: dict[str, Any], column: int, keep: set[str]) -> dict[str, Any]:
    table = payload["tables"][0]
    table["data"] = [row for row in table["data"] if row[column] in keep]
    return payload


@responses.activate
@pytest.mark.django_db
def test_attention_looks_back_one_session(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("tpex/disposition/attention_20261028_30.json")
    responses.post(ATTENTION, json=_within(payload, 5, {"115/10/29", "115/10/30"}))
    day = date(2026, 10, 30)
    events = _events(adapter, "attention", day, day)
    assert _sent(0) == {
        "startDate": "2026/10/29",
        "endDate": "2026/10/30",
        "type": "all",
        "code": "",
        "cate": "",
        "order": "date",
        "response": "json",
    }
    previous = date(2026, 10, 29)
    assert events == {
        ("1569", day, "attention", day),
        ("3324", day, "attention", day),
        ("700257", day, "attention", day),
        ("1569", previous, "attention", previous),
        ("54642", previous, "attention", previous),
    }
    assert RawTpex.objects.get().dataset == "attention"


@responses.activate
@pytest.mark.django_db
def test_attention_of_2010(adapter_2010: TpexAdapter, load_fixture: Callable[[str], Any]) -> None:
    responses.post(ATTENTION, json=load_fixture("tpex/disposition/attention_20100104_05.json"))
    events = _events(adapter_2010, "attention", date(2010, 1, 5), date(2010, 1, 5))
    assert ("3162", date(2010, 1, 4), "attention", date(2010, 1, 4)) in events
    assert len(events) == 3


@responses.activate
@pytest.mark.django_db
def test_attention_outside_the_query_is_a_schema_change(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(ATTENTION, json=load_fixture("tpex/disposition/attention_20261028_30.json"))
    day = date(2026, 10, 30)
    with pytest.raises(SchemaChangedError, match="1569"):
        adapter.fetch("attention", day, day)
    assert RawTpex.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_disposal_month_skips_placeholder_rows(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(DISPOSAL, json=load_fixture("tpex/disposition/disposal_202610.json"))
    events = _events(adapter, "disposition", *OCTOBER)
    assert _sent(0) == {
        "startDate": "2026/10/01",
        "endDate": "2026/10/30",
        "type": "all",
        "code": "",
        "cate": "",
        "reason": "-1",
        "measure": "-1",
        "order": "date",
        "response": "json",
    }
    assert ("4577", date(2026, 10, 30), "disposition", date(2026, 11, 5)) in events
    assert ("3441", date(2026, 10, 8), "disposition", date(2026, 10, 19)) in events
    assert ("6218", date(2026, 9, 23), "disposition", date(2026, 10, 5)) in events
    assert len(events) == 10


@responses.activate
@pytest.mark.django_db
def test_disposal_of_2010_with_placeholders_and_year_end(
    adapter_2010: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(DISPOSAL, json=load_fixture("tpex/disposition/disposal_201001.json"))
    events = _events(adapter_2010, "disposition", date(2010, 1, 1), date(2010, 1, 31))
    assert ("8277", date(2009, 12, 30), "disposition", date(2010, 1, 6)) in events
    assert len(events) == 6


@responses.activate
@pytest.mark.django_db
def test_row_without_a_code_that_is_not_a_placeholder_is_a_schema_change(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("tpex/disposition/disposal_202610.json")
    payload["tables"][0]["data"][0][2] = ""
    responses.post(DISPOSAL, json=payload)
    with pytest.raises(SchemaChangedError, match="without a code"):
        adapter.fetch("disposition", *OCTOBER)


@responses.activate
@pytest.mark.django_db
def test_full_delivery_includes_altered_and_managed_stocks(
    adapter_2010: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(CHTM, json=load_fixture("tpex/disposition/chtm_20100104.json"))
    day = date(2010, 1, 4)
    events = _events(adapter_2010, "full_delivery", day, day)
    assert _sent(0) == {"date": "2010/01/04", "response": "json"}
    assert events == {
        (code, day, "full_delivery", day) for code in ("1107", "2381", "3252", "4502")
    }


@responses.activate
@pytest.mark.django_db
def test_full_delivery_skips_suspended_only_rows(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(CHTM, json=load_fixture("tpex/disposition/chtm_20261028.json"))
    events = _events(adapter, "full_delivery", DAY, DAY)
    assert len(events) == 19
    assert "4804" not in {event[0] for event in events}


@responses.activate
@pytest.mark.django_db
def test_unknown_flag_is_a_schema_change(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("tpex/disposition/chtm_20261028.json")
    payload["tables"][0]["data"][0][2] = "Y"
    responses.post(CHTM, json=payload)
    with pytest.raises(SchemaChangedError, match="flag"):
        adapter.fetch("full_delivery", DAY, DAY)


@responses.activate
@pytest.mark.django_db
def test_full_delivery_of_the_previous_day_is_not_ready(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("tpex/disposition/chtm_20261028.json")
    responses.post(CHTM, json={**payload, "date": "20261027"})
    with pytest.raises(SourceNotReadyError, match="still serves 20261027"):
        adapter.fetch("full_delivery", DAY, DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("url", "dataset", "fixture", "change", "message"),
    [
        (
            DISPOSAL,
            "disposition",
            "disposal_202610.json",
            lambda t: t["fields"].__setitem__(5, "處置期間"),
            "處置起訖時間",
        ),
        (
            DISPOSAL,
            "disposition",
            "disposal_202610.json",
            lambda t: t["data"][0].__setitem__(5, "115/10/30"),
            "period",
        ),
        (
            ATTENTION,
            "attention",
            "attention_20261028_30.json",
            lambda t: t["fields"].__setitem__(5, "日期"),
            "公告日期",
        ),
        (CHTM, "full_delivery", "chtm_20261028.json", lambda t: t["data"].pop(), "rows"),
    ],
    ids=["period-renamed", "period-text", "date-renamed", "chtm-truncated"],
)
def test_layout_change_is_a_schema_change_and_keeps_raw(
    adapter: TpexAdapter,
    load_fixture: Callable[[str], Any],
    url: str,
    dataset: str,
    fixture: str,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    payload = load_fixture(f"tpex/disposition/{fixture}")
    change(payload["tables"][0])
    responses.post(url, json=payload)
    span = OCTOBER if dataset != "full_delivery" else (DAY, DAY)
    with pytest.raises(SchemaChangedError, match=message):
        adapter.fetch(dataset, *span)
    assert RawTpex.objects.count() == 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(("url", "dataset"), [(ATTENTION, "attention"), (DISPOSAL, "disposition")])
def test_empty_list_writes_nothing(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], url: str, dataset: str
) -> None:
    fixture = "attention_20261028_30.json" if dataset == "attention" else "disposal_202610.json"
    payload = load_fixture(f"tpex/disposition/{fixture}")
    payload["tables"][0]["data"] = []
    responses.post(url, json=payload)
    assert (
        adapter.upsert(dataset, adapter.normalize(dataset, adapter.fetch(dataset, DAY, DAY))) == 0
    )


@responses.activate
@pytest.mark.parametrize("dataset", ["attention", "disposition", "full_delivery"])
def test_non_trading_day_is_rejected_before_any_request(adapter: TpexAdapter, dataset: str) -> None:
    with pytest.raises(NotATradingDayError):
        adapter.fetch(dataset, date(2026, 10, 25), date(2026, 10, 25))
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_fetch_command_backfills_a_month_of_disposals_idempotently(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    payload = load_fixture("tpex/disposition/disposal_202610.json")
    responses.post(DISPOSAL, json=payload)
    for _ in range(2):
        call_command(
            "fetch", source="tpex", dataset="disposition", start="2026-10-01", end="2026-10-30"
        )
    assert len(responses.calls) == 2
    assert DispositionEvent.objects.count() == 10
    run = EtlRun.objects.filter(dataset="disposition").latest("run_id")
    assert (run.status, run.rows, run.checkpoint) == ("success", 10, {"last_date": "2026-10-30"})
    assert copy.deepcopy(payload) == RawTpex.objects.get().payload


@responses.activate
@pytest.mark.django_db
def test_unexpected_stat_is_a_schema_change_and_keeps_raw(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("tpex/disposition/disposal_202610.json")
    responses.post(DISPOSAL, json={**payload, "stat": "error"})
    with pytest.raises(SchemaChangedError, match="stat"):
        adapter.fetch("disposition", *OCTOBER)
    assert RawTpex.objects.count() == 1
