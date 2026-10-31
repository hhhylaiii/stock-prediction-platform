from collections.abc import Callable
from datetime import date
from typing import Any

import pytest

from apps.etl.adapters.base import SchemaChangedError
from apps.etl.trading_calendar import build_calendar, schedule_days


@pytest.fixture
def schedule_2026(load_fixture: Callable[[str], Any]) -> dict[str, Any]:
    payload: dict[str, Any] = load_fixture("twse/reference/holiday_schedule_2026.json")
    return payload


@pytest.fixture
def finmind_2010q1(load_fixture: Callable[[str], Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = load_fixture("finmind/trading_date_2010q1.json")["data"]
    return rows


def test_past_days_follow_the_traded_dates(finmind_2010q1: list[dict[str, Any]]) -> None:
    calendar, notes = build_calendar(finmind_2010q1, [], start=date(2010, 1, 1))
    assert (calendar.start, calendar.end) == (date(2010, 1, 1), date(2010, 3, 31))
    assert len(calendar.between(date(2010, 1, 1), date(2010, 3, 31))) == 57
    assert calendar.is_trading_day(date(2010, 2, 6))
    for closed in (date(2010, 1, 1), date(2010, 2, 11), date(2010, 2, 19), date(2010, 3, 6)):
        assert not calendar.is_trading_day(closed)
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
    past = [{"date": "2025-12-31"}]
    calendar, notes = build_calendar(past, [schedule_2026], start=date(2025, 12, 31))
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


def test_traded_dates_win_over_the_schedule(schedule_2026: dict[str, Any]) -> None:
    past = [{"date": d} for d in ("2026-10-22", "2026-10-27")]
    calendar, notes = build_calendar(past, [schedule_2026], start=date(2026, 10, 22))
    assert not calendar.is_trading_day(date(2026, 10, 23))
    assert not calendar.is_trading_day(date(2026, 10, 26))
    assert notes["2026-10-26"] == "臺灣光復暨金門古寧頭大捷紀念日"
    assert calendar.is_trading_day(date(2026, 10, 28))


def test_unpublished_year_ends_the_calendar(
    schedule_2026: dict[str, Any], load_fixture: Callable[[str], Any]
) -> None:
    empty_2027 = load_fixture("twse/reference/holiday_schedule_2027.json")
    calendar, _ = build_calendar(
        [{"date": "2026-10-30"}], [schedule_2026, empty_2027], start=date(2026, 10, 30)
    )
    assert calendar.end == date(2026, 12, 31)


def test_a_gap_before_the_first_schedule_year_is_refused(schedule_2026: dict[str, Any]) -> None:
    with pytest.raises(SchemaChangedError, match="2025"):
        build_calendar([{"date": "2024-12-31"}], [schedule_2026], start=date(2024, 12, 31))


def test_no_traded_dates_is_refused(schedule_2026: dict[str, Any]) -> None:
    with pytest.raises(SchemaChangedError, match="no trading dates"):
        build_calendar([], [schedule_2026], start=date(2026, 1, 1))


def test_traded_dates_before_start_are_ignored(finmind_2010q1: list[dict[str, Any]]) -> None:
    calendar, _ = build_calendar(finmind_2010q1, [], start=date(2010, 2, 1))
    assert calendar.start == date(2010, 2, 1)


@pytest.mark.parametrize(
    "change",
    [{"stat": "error"}, {"fields": ["日期", "名稱"]}],
)
def test_schedule_layout_changes_are_schema_changes(
    schedule_2026: dict[str, Any], change: dict[str, Any]
) -> None:
    with pytest.raises(SchemaChangedError):
        schedule_days({**schedule_2026, **change})


def test_rows_without_dates_are_schema_changes() -> None:
    with pytest.raises(SchemaChangedError, match="date"):
        build_calendar([{"day": "2026-10-30"}], [], start=date(2026, 10, 30))


@pytest.mark.parametrize("bad", ["2026/10/30", "", None])
def test_unparseable_traded_dates_are_schema_changes(bad: object) -> None:
    with pytest.raises(SchemaChangedError, match="bad date"):
        build_calendar([{"date": bad}], [], start=date(2026, 10, 1))


def test_unparseable_schedule_dates_are_schema_changes(schedule_2026: dict[str, Any]) -> None:
    broken = {**schedule_2026, "data": [["115/01/01", "中華民國開國紀念日", ""]]}
    with pytest.raises(SchemaChangedError, match="bad date"):
        schedule_days(broken)


def test_schedule_without_a_year_is_a_schema_change(schedule_2026: dict[str, Any]) -> None:
    broken = {key: value for key, value in schedule_2026.items() if key != "queryYear"}
    with pytest.raises(SchemaChangedError, match="queryYear"):
        build_calendar([{"date": "2025-12-31"}], [broken], start=date(2025, 12, 31))
