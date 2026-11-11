import subprocess
import sys
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlsplit
from zoneinfo import ZoneInfo

import pytest
import responses
import time_machine
from django.core.management import CommandError, call_command
from django.utils import timezone

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.adapters.twse import TwseAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import DataQualityLog, DividendNotice, EtlRun, RawTpex, RawTwse
from apps.etl.ratelimit import RateLimiter
from apps.etl.replay import replay_range
from apps.market.calendar import NotATradingDayError, TradingDays
from apps.market.models import DividendEvent, StockMaster

pytestmark = pytest.mark.django_db

TAIPEI = ZoneInfo("Asia/Taipei")
TWT48U = "https://www.twse.com.tw/rwd/zh/exRight/TWT48U"
PRE_POST = "https://www.tpex.org.tw/www/zh-tw/bulletin/prePost"
TWSE_LIST = "twse/dividend_notice/twt48u_20261111.json"
EARLIER_TWSE_LIST = "twse/dividend_notice/twt48u_20261107.json"
TPEX_LIST = "tpex/dividend_notice/prepost_20261111.json"
NO_DATA = {"stat": "很抱歉，沒有符合條件的資料!"}
NOV_11, NOV_12, NOV_13, NOV_16 = (date(2026, 11, day) for day in (11, 12, 13, 16))
START, END = date(2026, 10, 1), date(2027, 1, 29)
SPAN = (START + timedelta(days=offset) for offset in range((END - START).days + 1))
CALENDAR = TradingDays([day for day in SPAN if day.weekday() < 5], START, END)


def _at(day: date, clock: str) -> datetime:
    return datetime.fromisoformat(f"{day.isoformat()}T{clock}").replace(tzinfo=TAIPEI)


def _listed(*codes: str) -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code, name=code, market="TWSE") for code in codes
    )


def _announced() -> dict[tuple[str, date], date | None]:
    rows = DividendEvent.objects.values_list("stock_id", "ex_date", "announce_date")
    return {(code, ex_date): announced for code, ex_date, announced in rows}


@pytest.fixture
def twse(twse_limiter: RateLimiter) -> TwseAdapter:
    return TwseAdapter(CALENDAR, twse_limiter)


@pytest.fixture
def tpex(twse_limiter: RateLimiter) -> TpexAdapter:
    return TpexAdapter(CALENDAR, twse_limiter)


@responses.activate
@time_machine.travel(_at(NOV_11, "21:30:00"), tick=False)
def test_twse_notices_date_the_coming_events_of_listed_stocks(
    twse: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _listed("2344", "1437", "6655")
    responses.get(TWT48U, json=load_fixture(TWSE_LIST))

    run = twse.run("dividend_notice", NOV_11, NOV_11)

    assert (run.status, run.rows) == ("success", 3)
    assert _announced() == {
        ("2344", date(2026, 11, 18)): NOV_11,
        ("6655", date(2026, 11, 24)): NOV_11,
        ("1437", date(2026, 12, 3)): NOV_11,
    }
    assert dict(parse_qsl(urlsplit(responses.calls[0].request.url or "").query)) == {
        "response": "json"
    }
    assert RawTwse.objects.filter(dataset="dividend_notice").count() == 1
    assert DividendNotice.objects.filter(source="twse").count() == 49


@responses.activate
@time_machine.travel(_at(NOV_11, "21:30:00"), tick=False)
def test_tpex_notices_date_the_coming_events_of_listed_stocks(
    tpex: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _listed("3680", "6488", "5483", "3289")
    responses.post(PRE_POST, json=load_fixture(TPEX_LIST))

    run = tpex.run("dividend_notice", NOV_11, NOV_11)

    assert (run.status, run.rows) == ("success", 3)
    assert _announced() == {
        ("3680", date(2026, 11, 12)): NOV_11,
        ("6488", date(2026, 12, 29)): NOV_11,
        ("5483", date(2027, 1, 5)): NOV_11,
    }
    body = responses.calls[0].request.body or ""
    sent = parse_qs(body.decode() if isinstance(body, bytes) else body)
    assert sent == {"response": ["json"]}
    assert RawTpex.objects.filter(dataset="dividend_notice").count() == 1


@pytest.mark.parametrize(
    ("day", "moment", "seen"),
    [
        (NOV_11, _at(NOV_11, "22:39:59"), NOV_11),
        (NOV_11, _at(NOV_11, "22:40:00"), NOV_12),
        (NOV_11, _at(NOV_12, "00:30:00"), NOV_12),
        (NOV_13, _at(date(2026, 11, 14), "10:00:00"), NOV_16),
    ],
)
@responses.activate
def test_the_moment_of_the_fetch_decides_the_session_not_the_day_asked(
    twse: TwseAdapter, load_fixture: Callable[[str], Any], day: date, moment: datetime, seen: date
) -> None:
    _listed("6655")
    responses.get(TWT48U, json=load_fixture(TWSE_LIST))

    with time_machine.travel(moment, tick=False):
        twse.run("dividend_notice", day, day)

    assert _announced() == {("6655", date(2026, 11, 24)): seen}


@responses.activate
def test_an_unchanged_list_is_kept_once_for_each_session_that_read_it(
    twse: TwseAdapter, tpex: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(TWT48U, json=load_fixture(TWSE_LIST))
    responses.post(PRE_POST, json=load_fixture(TPEX_LIST))

    for day, clock in ((NOV_11, "21:30:00"), (NOV_11, "22:00:00"), (NOV_12, "21:30:00")):
        with time_machine.travel(_at(day, clock), tick=False):
            twse.fetch("dividend_notice", day, day)
            tpex.fetch("dividend_notice", day, day)

    for raw_model in (RawTwse, RawTpex):
        kept = raw_model.objects.filter(dataset="dividend_notice").order_by("fetched_at")
        assert [timezone.localtime(row.fetched_at).date() for row in kept] == [NOV_11, NOV_12]
        assert {str(row.request_params) for row in kept} == {"{'response': 'json'}"}


@responses.activate
@time_machine.travel(_at(NOV_11, "21:30:00"), tick=False)
def test_a_list_with_nothing_on_it_has_not_been_published(
    twse: TwseAdapter, tpex: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(TWT48U, json=NO_DATA)
    empty = load_fixture(TPEX_LIST)
    empty["tables"][0].update(data=[], totalCount=0)
    responses.post(PRE_POST, json=empty)

    with pytest.raises(SourceNotReadyError, match="TWT48U"):
        twse.fetch("dividend_notice", NOV_11, NOV_11)
    with pytest.raises(SourceNotReadyError, match="prePost"):
        tpex.fetch("dividend_notice", NOV_11, NOV_11)


@responses.activate
@time_machine.travel(_at(NOV_13, "21:30:00"), tick=False)
def test_a_span_of_days_is_refused_since_the_list_has_no_history(
    twse: TwseAdapter, tpex: TpexAdapter
) -> None:
    for adapter in (twse, tpex):
        with pytest.raises(ValueError, match="one day"):
            adapter.run("dividend_notice", NOV_11, NOV_13)
        with pytest.raises(ValueError, match="one day"):
            adapter.fetch("dividend_notice", NOV_11, NOV_13)
    assert len(responses.calls) == 0
    assert not EtlRun.objects.exists()


@responses.activate
@time_machine.travel(_at(date(2026, 11, 14), "10:00:00"), tick=False)
def test_a_day_the_market_is_closed_is_refused(twse: TwseAdapter, tpex: TpexAdapter) -> None:
    saturday = date(2026, 11, 14)

    for adapter in (twse, tpex):
        with pytest.raises(NotATradingDayError):
            adapter.fetch("dividend_notice", saturday, saturday)
    assert len(responses.calls) == 0


@responses.activate
@time_machine.travel(_at(NOV_11, "21:30:00"), tick=False)
def test_an_answer_that_is_not_the_list_is_refused_and_still_kept(
    twse: TwseAdapter, tpex: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(TWT48U, json={"stat": "查詢日期小於可查詢最小日期"})
    odd = load_fixture(TPEX_LIST)
    odd["stat"] = "error"
    responses.post(PRE_POST, json=odd)

    with pytest.raises(SchemaChangedError, match="TWT48U"):
        twse.fetch("dividend_notice", NOV_11, NOV_11)
    with pytest.raises(SchemaChangedError, match="prePost"):
        tpex.fetch("dividend_notice", NOV_11, NOV_11)
    assert RawTwse.objects.count() == RawTpex.objects.count() == 1


@time_machine.travel(_at(NOV_13, "21:30:00"), tick=False)
def test_reprocess_refuses_the_list_which_no_date_keys() -> None:
    with pytest.raises(ValueError, match="reapply_dividend_notices"):
        replay_range("twse", "dividend_notice", NOV_11, NOV_11, CALENDAR)


@time_machine.travel(_at(NOV_13, "21:30:00"), tick=False)
def test_stored_lists_apply_again_in_the_order_and_session_they_were_fetched(
    load_fixture: Callable[[str], Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _listed("2344", "6655", "00401A")
    refused = {**load_fixture(TWSE_LIST), "stat": "查詢日期小於可查詢最小日期"}
    kept = (
        ("b", _at(NOV_11, "21:30:00"), load_fixture(TWSE_LIST)),
        ("a", _at(date(2026, 11, 7), "10:00:00"), load_fixture(EARLIER_TWSE_LIST)),
        ("c", _at(NOV_12, "21:30:00"), NO_DATA),
        ("d", _at(NOV_12, "22:00:00"), refused),
    )
    for tag, moment, payload in kept:
        RawTwse.objects.create(
            dataset="dividend_notice",
            source_hash=tag,
            request_params={"response": "json"},
            payload=payload,
            fetched_at=moment,
        )
    monkeypatch.setattr(TradingDays, "from_db", classmethod(lambda cls: CALENDAR))
    alerted: list[object] = []
    monkeypatch.setattr("apps.etl.dividend_notices.dispatch", alerted.append)

    call_command("reapply_dividend_notices")
    first = capsys.readouterr().out
    call_command("reapply_dividend_notices", "--source", "twse")

    assert "twse: 2 lists read, 2 rows written, 2 skipped" in first
    assert "tpex: 0 lists read, 0 rows written, 0 skipped" in first
    assert "twse: 2 lists read, 0 rows written, 2 skipped" in capsys.readouterr().out
    assert alerted == []
    assert _announced() == {
        ("2344", date(2026, 11, 18)): date(2026, 11, 9),
        ("6655", date(2026, 11, 24)): NOV_11,
    }


@responses.activate
@time_machine.travel(_at(NOV_11, "21:30:00"), tick=False)
def test_a_preview_prints_what_the_lists_would_write_and_keeps_nothing(
    load_fixture: Callable[[str], Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _listed("2344", "6655", "0051", "6488")
    DividendEvent.objects.create(stock_id="6655", ex_date=date(2026, 11, 24), adj_factor=0.97)
    responses.get(TWT48U, json=load_fixture(TWSE_LIST))
    responses.post(PRE_POST, json=load_fixture(TPEX_LIST))
    monkeypatch.setattr(TradingDays, "from_db", classmethod(lambda cls: CALENDAR))
    alerted: list[object] = []
    monkeypatch.setattr("apps.etl.dividend_notices.dispatch", alerted.append)

    call_command("preview_dividend_notices")

    out = capsys.readouterr().out
    assert alerted == []
    assert "twse session 2026-11-11: 49 listed, sound, 2 to create, 1 to date" in out
    assert "  create 2344 2026-11-18 cash 2.00000000 stock 0.00000000 announced 2026-11-11" in out
    assert "  create 0051 2026-11-20 cash 0.00000000 stock 0.00000000 announced 2026-11-11" in out
    assert "  date 6655 2026-11-24 announced 2026-11-11" in out
    assert "  create 6488 2026-12-29 cash 6.00000000 stock 0.00000000 announced 2026-11-11" in out
    assert out.rstrip().endswith("nothing was written")
    assert _announced() == {("6655", date(2026, 11, 24)): None}
    kept = (DividendNotice, RawTwse, RawTpex, DataQualityLog, EtlRun)
    assert [model.objects.count() for model in kept] == [0, 0, 0, 0, 0]


@responses.activate
@time_machine.travel(_at(NOV_11, "21:30:00"), tick=False)
def test_a_preview_of_one_source_that_has_not_published_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses.get(TWT48U, json=NO_DATA)
    monkeypatch.setattr(TradingDays, "from_db", classmethod(lambda cls: CALENDAR))

    with pytest.raises(CommandError, match="SourceNotReadyError"):
        call_command("preview_dividend_notices", "--source", "twse")


@pytest.mark.parametrize("command", ["preview_dividend_notices", "reapply_dividend_notices"])
def test_the_notice_commands_load_in_a_fresh_process(command: str) -> None:
    root = Path(__file__).resolve().parents[2]
    asked = [sys.executable, "manage.py", command, "--help"]
    result = subprocess.run(asked, cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-400:]
