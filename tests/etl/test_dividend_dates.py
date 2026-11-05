from collections.abc import Callable
from datetime import date, time, timedelta
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


def test_the_earliest_announcement_of_an_ex_date_counts(esun: Rows) -> None:
    rows = policy(esun, AnnouncementDate="2013-07-20") + policy(esun)
    assert ex_date_announcements(rows, "2884", SESSION) == {date(2013, 8, 15): date(2013, 7, 22)}


def test_an_announcement_after_the_ex_date_gives_no_date(esun: Rows) -> None:
    found = ex_date_announcements(policy(esun, AnnouncementDate="2013-08-20"), "2884", SESSION)
    assert found == {date(2013, 8, 15): None}


def test_a_bad_announcement_time_is_refused(esun: Rows) -> None:
    with pytest.raises(SchemaChangedError, match="bad time"):
        ex_date_announcements(policy(esun, AnnouncementTime="8pm"), "2884", SESSION)


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
