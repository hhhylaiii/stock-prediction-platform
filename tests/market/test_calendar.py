import json
import re
from datetime import date
from pathlib import Path

import pytest

from apps.market.calendar import (
    CalendarError,
    NotATradingDayError,
    OutsideCalendarError,
    TradingDays,
)
from apps.market.models import TradingCalendar
from tests.conftest import FIXTURES

REPO_ROOT = Path(__file__).resolve().parents[2]
BANNED_CALENDAR_CALLS = re.compile(
    r"bdate_range|\bBDay\b|BusinessDay|CustomBusinessDay|busday_offset|busday_count"
)


def d(text: str) -> date:
    return date.fromisoformat(text)


@pytest.mark.parametrize(
    ("start", "n", "expected"),
    [
        ("2015-09-25", 1, "2015-09-30"),
        ("2015-09-30", -1, "2015-09-25"),
        ("2016-09-26", 1, "2016-09-29"),
        ("2016-07-07", 1, "2016-07-11"),
        ("2016-06-03", 1, "2016-06-04"),
        ("2016-06-04", 1, "2016-06-06"),
        ("2014-12-26", 1, "2014-12-27"),
        ("2015-05-29", 1, "2015-06-01"),
        ("2016-12-30", 0, "2016-12-30"),
    ],
)
def test_shift_follows_holidays_typhoons_and_makeup_saturdays(
    twse_calendar: TradingDays, start: str, n: int, expected: str
) -> None:
    assert twse_calendar.shift(d(start), n) == d(expected)


def test_shift_round_trips_over_many_sessions(twse_calendar: TradingDays) -> None:
    start = d("2015-01-05")
    assert twse_calendar.shift(twse_calendar.shift(start, 250), -250) == start


@pytest.mark.parametrize("day", ["2015-09-29", "2016-06-05", "2015-09-28"])
def test_shift_rejects_non_trading_day(twse_calendar: TradingDays, day: str) -> None:
    with pytest.raises(NotATradingDayError):
        twse_calendar.shift(d(day), 1)


def test_shift_past_either_end_is_rejected(twse_calendar: TradingDays) -> None:
    with pytest.raises(OutsideCalendarError):
        twse_calendar.shift(d("2016-12-30"), 1)
    with pytest.raises(OutsideCalendarError):
        twse_calendar.shift(d("2014-01-02"), -1)


def test_dates_outside_range_are_rejected(twse_calendar: TradingDays) -> None:
    with pytest.raises(OutsideCalendarError):
        twse_calendar.is_trading_day(d("2017-01-03"))
    with pytest.raises(OutsideCalendarError):
        twse_calendar.between(d("2013-12-31"), d("2014-01-10"))


def test_is_trading_day_flags(twse_calendar: TradingDays) -> None:
    assert twse_calendar.is_trading_day(d("2015-06-01"))
    assert twse_calendar.is_trading_day(d("2016-06-04"))
    assert not twse_calendar.is_trading_day(d("2015-09-29"))
    assert not twse_calendar.is_trading_day(d("2016-06-05"))


def test_between_is_inclusive_and_skips_closures(twse_calendar: TradingDays) -> None:
    assert twse_calendar.between(d("2015-09-25"), d("2015-09-30")) == (
        d("2015-09-25"),
        d("2015-09-30"),
    )
    assert twse_calendar.between(d("2015-09-28"), d("2015-09-29")) == ()


def test_between_rejects_reversed_range(twse_calendar: TradingDays) -> None:
    with pytest.raises(CalendarError):
        twse_calendar.between(d("2015-09-30"), d("2015-09-25"))


def test_latest_on_or_before(twse_calendar: TradingDays) -> None:
    assert twse_calendar.latest_on_or_before(d("2015-09-29")) == d("2015-09-25")
    assert twse_calendar.latest_on_or_before(d("2015-09-30")) == d("2015-09-30")
    with pytest.raises(OutsideCalendarError):
        twse_calendar.latest_on_or_before(d("2014-01-01"))


def test_fixture_session_counts_match_twse(twse_calendar: TradingDays) -> None:
    counts = {
        year: len(twse_calendar.between(date(year, 1, 1), date(year, 12, 31)))
        for year in (2014, 2015, 2016)
    }
    assert counts == {2014: 248, 2015: 244, 2016: 244}


def test_constructor_rejects_days_outside_range() -> None:
    with pytest.raises(CalendarError):
        TradingDays([d("2015-01-05")], d("2015-02-01"), d("2015-02-28"))
    with pytest.raises(CalendarError):
        TradingDays([], d("2015-02-28"), d("2015-02-01"))


@pytest.mark.django_db
def test_from_db_matches_fixture(twse_calendar: TradingDays) -> None:
    notes = json.loads((FIXTURES / "calendar" / "twse_2014_2016.json").read_text())["notes"]
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=is_open, note=notes.get(day.isoformat(), ""))
        for day, is_open in twse_calendar.all_days()
    )
    loaded = TradingDays.from_db()
    assert (loaded.start, loaded.end) == (d("2014-01-01"), d("2016-12-31"))
    assert loaded.between(loaded.start, loaded.end) == twse_calendar.between(
        twse_calendar.start, twse_calendar.end
    )
    assert TradingCalendar.objects.get(date=d("2015-09-29")).note.startswith("颱風假")


@pytest.mark.django_db
def test_from_db_rejects_gaps() -> None:
    TradingCalendar.objects.create(date=d("2015-01-05"), is_trading_day=True)
    TradingCalendar.objects.create(date=d("2015-01-07"), is_trading_day=True)
    with pytest.raises(CalendarError, match="missing"):
        TradingDays.from_db()


@pytest.mark.django_db
def test_from_db_rejects_empty_table() -> None:
    with pytest.raises(CalendarError, match="empty"):
        TradingDays.from_db()


def test_no_business_day_helpers_outside_tests() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for top in ("apps", "config", "scripts")
        for path in (REPO_ROOT / top).rglob("*.py")
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if BANNED_CALENDAR_CALLS.search(line)
    ]
    assert offenders == []
