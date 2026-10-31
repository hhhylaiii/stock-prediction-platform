from collections.abc import Callable
from datetime import date
from typing import Any

import pytest

from apps.etl.adapters.base import SchemaChangedError
from apps.etl.trading_calendar import build_calendar, schedule_days, traded_days


@pytest.fixture
def schedule_2026(load_fixture: Callable[[str], Any]) -> dict[str, Any]:
    payload: dict[str, Any] = load_fixture("twse/reference/holiday_schedule_2026.json")
    return payload


@pytest.fixture
def february_2010(load_fixture: Callable[[str], Any]) -> set[date]:
    return traded_days(load_fixture("twse/reference/fmtqik_201002.json"), date(2010, 2, 1))


def test_monthly_table_lists_the_trading_days(february_2010: set[date]) -> None:
    assert len(february_2010) == 14
    assert date(2010, 2, 6) in february_2010
    assert date(2010, 2, 11) not in february_2010


def test_an_unpublished_month_is_empty_only_while_it_is_current(
    load_fixture: Callable[[str], Any],
) -> None:
    empty = load_fixture("twse/reference/fmtqik_202611.json")
    assert traded_days(empty, date(2026, 11, 1), allow_empty=True) == set()
    with pytest.raises(SchemaChangedError, match="2026-11"):
        traded_days(empty, date(2026, 11, 1))


@pytest.mark.parametrize(
    "change",
    [
        {"stat": "error"},
        {"fields": ["成交股數", "日期"]},
        {"data": [["115/10/01", "1", "1", "1", "1", "1"]]},
        {"data": [["10/01", "1", "1", "1", "1", "1"]]},
    ],
)
def test_monthly_table_changes_are_schema_changes(
    load_fixture: Callable[[str], Any], change: dict[str, Any]
) -> None:
    payload = {**load_fixture("twse/reference/fmtqik_201002.json"), **change}
    with pytest.raises(SchemaChangedError):
        traded_days(payload, date(2010, 2, 1))


def test_past_days_follow_the_traded_days(february_2010: set[date]) -> None:
    calendar, notes = build_calendar(february_2010, date(2010, 2, 28), [], start=date(2010, 2, 1))
    assert (calendar.start, calendar.end) == (date(2010, 2, 1), date(2010, 2, 28))
    assert len(calendar.between(date(2010, 2, 1), date(2010, 2, 28))) == 14
    assert calendar.is_trading_day(date(2010, 2, 6))
    assert not calendar.is_trading_day(date(2010, 2, 12))
    assert notes == {}


@pytest.mark.parametrize(
    ("day", "trading", "note"),
    [
        (date(2026, 1, 1), False, "中華民國開國紀念日"),
        (date(2026, 1, 2), True, ""),
        (date(2026, 2, 11), True, ""),
        (date(2026, 2, 12), False, "市場無交易，僅辦理結算交割作業"),
        (date(2026, 2, 20), False, "農曆除夕及春節"),
        (date(2026, 2, 23), True, ""),
        (date(2026, 2, 28), False, "和平紀念日"),
        (date(2026, 11, 2), True, ""),
        (date(2026, 11, 7), False, ""),
        (date(2026, 12, 25), False, "行憲紀念日"),
        (date(2026, 12, 31), True, ""),
    ],
)
def test_future_days_follow_the_holiday_schedule(
    schedule_2026: dict[str, Any], day: date, trading: bool, note: str
) -> None:
    calendar, notes = build_calendar(
        {date(2025, 12, 31)}, date(2025, 12, 31), [schedule_2026], start=date(2025, 12, 31)
    )
    assert calendar.end == date(2026, 12, 31)
    assert calendar.is_trading_day(day) is trading
    assert notes.get(day.isoformat(), "") == note


def test_settlement_only_days_are_closed_even_when_named_a_trading_day(
    load_fixture: Callable[[str], Any],
) -> None:
    days = schedule_days(load_fixture("twse/reference/holiday_schedule_2021.json"))
    assert days[date(2021, 2, 5)] == (True, "")
    assert days[date(2021, 2, 8)] == (False, "農曆春節前最後交易日")
    assert days[date(2021, 2, 9)][0] is False
    assert days[date(2021, 1, 4)] == (True, "")


def test_traded_days_win_over_the_schedule(schedule_2026: dict[str, Any]) -> None:
    traded = {date(2026, 10, 22), date(2026, 10, 27)}
    calendar, notes = build_calendar(
        traded, date(2026, 10, 27), [schedule_2026], start=date(2026, 10, 22)
    )
    assert not calendar.is_trading_day(date(2026, 10, 23))
    assert not calendar.is_trading_day(date(2026, 10, 26))
    assert notes["2026-10-26"] == "臺灣光復暨金門古寧頭大捷紀念日"
    assert calendar.is_trading_day(date(2026, 10, 28))


def test_a_closure_right_before_today_is_kept(schedule_2026: dict[str, Any]) -> None:
    calendar, _ = build_calendar(
        {date(2026, 10, 21)}, date(2026, 10, 23), [schedule_2026], start=date(2026, 10, 21)
    )
    assert not calendar.is_trading_day(date(2026, 10, 22))
    assert not calendar.is_trading_day(date(2026, 10, 23))


def test_unpublished_year_ends_the_calendar(
    schedule_2026: dict[str, Any], load_fixture: Callable[[str], Any]
) -> None:
    empty_2027 = load_fixture("twse/reference/holiday_schedule_2027.json")
    calendar, _ = build_calendar(
        {date(2026, 10, 30)},
        date(2026, 10, 31),
        [schedule_2026, empty_2027],
        start=date(2026, 10, 30),
    )
    assert calendar.end == date(2026, 12, 31)


def test_a_gap_before_the_first_schedule_year_is_refused(schedule_2026: dict[str, Any]) -> None:
    with pytest.raises(SchemaChangedError, match="2025"):
        build_calendar(
            {date(2024, 12, 31)}, date(2024, 12, 31), [schedule_2026], start=date(2024, 12, 31)
        )


def test_traded_days_outside_the_range_are_ignored(february_2010: set[date]) -> None:
    calendar, _ = build_calendar(february_2010, date(2010, 2, 10), [], start=date(2010, 2, 5))
    assert (calendar.start, calendar.end) == (date(2010, 2, 5), date(2010, 2, 10))
    assert len(calendar.between(date(2010, 2, 5), date(2010, 2, 10))) == 5


@pytest.mark.parametrize(
    "change",
    [{"stat": "error"}, {"fields": ["日期", "名稱"]}],
)
def test_schedule_layout_changes_are_schema_changes(
    schedule_2026: dict[str, Any], change: dict[str, Any]
) -> None:
    with pytest.raises(SchemaChangedError):
        schedule_days({**schedule_2026, **change})


def test_unparseable_schedule_dates_are_schema_changes(schedule_2026: dict[str, Any]) -> None:
    broken = {**schedule_2026, "data": [["115/01/01", "中華民國開國紀念日", ""]]}
    with pytest.raises(SchemaChangedError, match="bad date"):
        schedule_days(broken)


def test_schedule_without_a_year_is_a_schema_change(schedule_2026: dict[str, Any]) -> None:
    broken = {key: value for key, value in schedule_2026.items() if key != "queryYear"}
    with pytest.raises(SchemaChangedError, match="queryYear"):
        build_calendar({date(2025, 12, 31)}, date(2025, 12, 31), [broken], start=date(2025, 12, 31))
