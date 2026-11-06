import subprocess
import sys
from collections.abc import Callable
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command

from apps.etl.adapters.finmind_datasets import dividend_events, ex_date_announcements
from apps.etl.dividend_dates import announcement_session, fill_announce_dates
from apps.etl.errors import SchemaChangedError
from apps.etl.models import RawFinmind
from apps.market.calendar import TradingDays
from apps.market.models import DividendEvent

Rows = list[dict[str, Any]]
CUTOFF = time(22, 40)


def weekdays(start: date, end: date) -> TradingDays:
    span = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return TradingDays([day for day in span if day.weekday() < 5], start, end)


CALENDAR = weekdays(date(2010, 1, 1), date(2027, 12, 31))
SESSION = announcement_session(CALENDAR, CUTOFF)


@pytest.fixture
def esun(load_fixture: Callable[[str], Any]) -> Rows:
    data: Rows = load_fixture("finmind/dividend_2884.json")["data"]
    return data


def policy(rows: Rows, **changes: str) -> Rows:
    return [{**row, **changes} for row in rows if row["CashExDividendTradingDate"] == "2013-08-15"]


def test_an_evening_announcement_before_the_cutoff_counts_that_session(esun: Rows) -> None:
    assert ex_date_announcements(policy(esun), "2884", SESSION) == {
        date(2013, 8, 15): date(2013, 7, 31)
    }


@pytest.mark.parametrize(
    ("clock", "seen"),
    [("22:39:59", date(2013, 7, 31)), ("22:40:00", date(2013, 8, 1)), ("", date(2013, 8, 1))],
)
def test_a_late_or_untimed_announcement_counts_the_next_session(
    esun: Rows, clock: str, seen: date
) -> None:
    found = ex_date_announcements(policy(esun, AnnouncementTime=clock), "2884", SESSION)
    assert found == {date(2013, 8, 15): seen}


def test_a_weekend_announcement_counts_the_next_session(esun: Rows) -> None:
    found = ex_date_announcements(policy(esun, AnnouncementDate="2013-08-03"), "2884", SESSION)
    assert found == {date(2013, 8, 15): date(2013, 8, 5)}


@pytest.mark.parametrize("earliest_first", [True, False])
def test_the_earliest_announcement_of_an_ex_date_counts(esun: Rows, earliest_first: bool) -> None:
    early, late = policy(esun, AnnouncementDate="2013-07-20"), policy(esun)
    rows = early + late if earliest_first else late + early
    assert ex_date_announcements(rows, "2884", SESSION) == {date(2013, 8, 15): date(2013, 7, 22)}


def test_an_announcement_in_the_ex_date_session_still_dates_it(esun: Rows) -> None:
    rows = policy(esun, AnnouncementDate="2013-08-15", AnnouncementTime="09:00:00")
    assert ex_date_announcements(rows, "2884", SESSION) == {date(2013, 8, 15): date(2013, 8, 15)}


def test_a_row_without_ex_dates_needs_no_session(esun: Rows) -> None:
    rows = policy(esun, AnnouncementDate="2028-01-03", CashExDividendTradingDate="")
    rows = [{**row, "StockExDividendTradingDate": ""} for row in rows]
    assert ex_date_announcements(rows, "2884", SESSION) == {}


def test_an_announcement_after_the_ex_date_gives_no_date(esun: Rows) -> None:
    found = ex_date_announcements(policy(esun, AnnouncementDate="2013-08-20"), "2884", SESSION)
    assert found == {date(2013, 8, 15): None}


def test_a_bad_announcement_time_is_refused(esun: Rows) -> None:
    with pytest.raises(SchemaChangedError, match="bad time"):
        ex_date_announcements(policy(esun, AnnouncementTime="8pm"), "2884", SESSION)


def test_answers_without_announcement_dates_are_refused(esun: Rows) -> None:
    rows = [{k: v for k, v in row.items() if k != "AnnouncementDate"} for row in policy(esun)]
    with pytest.raises(SchemaChangedError, match="AnnouncementDate"):
        ex_date_announcements(rows, "2884", SESSION)


def test_an_announcement_past_the_calendar_rejects_the_stock(esun: Rows) -> None:
    rows = policy(esun, AnnouncementDate="2028-01-03", CashExDividendTradingDate="2028-02-01")
    rows = [{**row, "StockExDividendTradingDate": "2028-02-01"} for row in rows]
    with pytest.raises(SchemaChangedError, match="calendar"):
        ex_date_announcements(rows, "2884", SESSION)


def test_events_carry_the_date_only_when_a_session_rule_is_given(esun: Rows) -> None:
    dated = dividend_events(policy(esun), [], "2884", session=SESSION).iloc[0]
    assert dated["announce_date"] == date(2013, 7, 31)
    assert dividend_events(policy(esun), [], "2884").iloc[0]["announce_date"] is None


@pytest.mark.django_db
def test_only_blank_dates_are_filled() -> None:
    DividendEvent.objects.create(stock_id="2884", ex_date=date(2013, 8, 15))
    DividendEvent.objects.create(
        stock_id="2884", ex_date=date(2014, 8, 14), announce_date=date(2014, 7, 1)
    )
    announced = {date(2013, 8, 15): date(2013, 7, 31), date(2014, 8, 14): date(2014, 7, 30)}
    assert fill_announce_dates("2884", announced) == 1
    dates = dict(DividendEvent.objects.values_list("ex_date", "announce_date"))
    assert dates == {date(2013, 8, 15): date(2013, 7, 31), date(2014, 8, 14): date(2014, 7, 1)}


@pytest.mark.django_db
def test_the_command_reads_stored_answers_again(
    esun: Rows, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    DividendEvent.objects.create(stock_id="2884", ex_date=date(2013, 8, 15))
    DividendEvent.objects.create(stock_id="2330", ex_date=date(2024, 3, 18))
    RawFinmind.objects.create(
        dataset="dividend_policy",
        source_hash="a",
        request_params={"dataset": "TaiwanStockDividend", "data_id": "2884"},
        payload={"data": policy(esun)},
    )
    RawFinmind.objects.create(
        dataset="dividend_policy",
        source_hash="b",
        request_params={"dataset": "TaiwanStockDividend", "data_id": "2330"},
        payload={"data": [{**policy(esun)[0], "stock_id": "2330", "AnnouncementTime": "x"}]},
    )
    monkeypatch.setattr(TradingDays, "from_db", classmethod(lambda cls: CALENDAR))
    call_command("reapply_dividend_dates")
    out, err = capsys.readouterr()
    assert "1 events dated across 2 stocks" in out
    assert "rejected 2330: SchemaChangedError" in err
    assert DividendEvent.objects.get(stock_id="2884").announce_date == date(2013, 7, 31)


@pytest.mark.django_db
def test_the_command_can_take_only_the_stocks_named(
    esun: Rows, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for code in ("2884", "2885"):
        DividendEvent.objects.create(stock_id=code, ex_date=date(2013, 8, 15))
        RawFinmind.objects.create(
            dataset="dividend_policy",
            source_hash=code,
            request_params={"dataset": "TaiwanStockDividend", "data_id": code},
            payload={"data": [{**row, "stock_id": code} for row in policy(esun)]},
        )
    monkeypatch.setattr(TradingDays, "from_db", classmethod(lambda cls: CALENDAR))
    call_command("reapply_dividend_dates", "--stock", "2885")
    assert "1 events dated across 1 stocks" in capsys.readouterr().out
    dates = dict(DividendEvent.objects.values_list("stock_id", "announce_date"))
    assert dates == {"2884": None, "2885": date(2013, 7, 31)}


@pytest.mark.django_db
def test_every_stored_answer_of_a_stock_is_read_and_an_empty_one_skipped(
    esun: Rows, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    later = [row for row in esun if row["CashExDividendTradingDate"] == "2014-08-21"]
    for ex in (date(2013, 8, 15), date(2014, 8, 21)):
        DividendEvent.objects.create(stock_id="2884", ex_date=ex)
    for tag, data in (("a", policy(esun)), ("b", later), ("c", None)):
        RawFinmind.objects.create(
            dataset="dividend_policy",
            source_hash=tag,
            request_params={"dataset": "TaiwanStockDividend", "data_id": "2884"},
            payload={"data": data},
        )
    monkeypatch.setattr(TradingDays, "from_db", classmethod(lambda cls: CALENDAR))
    call_command("reapply_dividend_dates")
    assert "2 events dated across 1 stocks" in capsys.readouterr().out
    dates = dict(DividendEvent.objects.values_list("ex_date", "announce_date"))
    assert dates == {date(2013, 8, 15): date(2013, 7, 31), date(2014, 8, 21): date(2014, 8, 6)}


def test_the_command_loads_in_a_fresh_process() -> None:
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "manage.py", "reapply_dividend_dates", "--help"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-400:]
