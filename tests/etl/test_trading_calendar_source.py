from collections.abc import Callable
from datetime import date
from typing import Any

import pytest

from apps.etl.errors import SchemaChangedError
from apps.etl.trading_calendar import (
    UNLABELLED_CLOSURE,
    WEEKEND_TRADING,
    build_calendar,
    require_whole_year,
    schedule_days,
    schedule_years,
    traded_days,
)


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
    assert notes["2010-02-12"] == UNLABELLED_CLOSURE


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
    assert days[date(2021, 2, 8)] == (False, "市場無交易，僅辦理結算交割作業")
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


def test_past_closures_take_schedule_notes_or_an_unlabelled_note(
    schedule_2026: dict[str, Any], load_fixture: Callable[[str], Any]
) -> None:
    october = traded_days(load_fixture("twse/reference/fmtqik_202610.json"), date(2026, 10, 1))
    calendar, notes = build_calendar(
        october, date(2026, 10, 31), [schedule_2026], start=date(2026, 10, 1)
    )
    assert notes["2026-10-09"] == "國慶日"
    assert notes["2026-10-23"] == UNLABELLED_CLOSURE
    assert "2026-10-24" not in notes
    assert calendar.end == date(2026, 12, 31)


def test_weekend_trading_days_are_noted(february_2010: set[date]) -> None:
    _, notes = build_calendar(february_2010, date(2010, 2, 28), [], start=date(2010, 2, 1))
    assert notes["2010-02-06"] == WEEKEND_TRADING
    assert notes["2010-02-11"] == UNLABELLED_CLOSURE


@pytest.mark.parametrize(
    ("start", "today", "years"),
    [
        (date(2010, 1, 1), date(2026, 11, 2), list(range(2021, 2028))),
        (date(2026, 10, 1), date(2026, 11, 2), [2026, 2027]),
        (date(2026, 12, 30), date(2027, 1, 1), [2026, 2027, 2028]),
    ],
)
def test_schedules_cover_the_noted_years_and_next_year(
    start: date, today: date, years: list[int]
) -> None:
    assert list(schedule_years(start, today)) == years


def test_an_empty_schedule_for_a_covered_year_is_a_schema_change(
    load_fixture: Callable[[str], Any],
) -> None:
    empty = {**load_fixture("twse/reference/holiday_schedule_2027.json"), "queryYear": 2026}
    with pytest.raises(SchemaChangedError, match="2026"):
        build_calendar({date(2026, 10, 30)}, date(2026, 10, 31), [empty], start=date(2026, 10, 30))


PUBLISHED_2027 = "twse/reference/holiday_schedule_2027_published.json"
SETTLEMENT_ONLY = "市場無交易，僅辦理結算交割作業"
SCHEDULED_SESSIONS_2023 = 240


def _last_day_row(detail: str) -> list[list[str]]:
    return [["2027-02-01", "農曆春節前最後交易日", detail]]


def test_a_last_trading_day_naming_the_closures_after_it_stays_open(
    load_fixture: Callable[[str], Any],
) -> None:
    days = schedule_days(load_fixture(PUBLISHED_2027))
    assert days[date(2027, 2, 1)] == (True, "")
    assert days[date(2027, 2, 2)] == (False, SETTLEMENT_ONLY)
    assert days[date(2027, 2, 3)] == (False, SETTLEMENT_ONLY)
    assert days[date(2027, 2, 4)] == (False, "農曆除夕及春節")
    assert days[date(2027, 2, 11)] == (True, "")


def test_the_published_2027_schedule_trades_on_242_days(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture(PUBLISHED_2027)
    calendar, notes = build_calendar((), date(2026, 12, 31), [payload], start=date(2027, 1, 1))
    sessions = calendar.between(date(2027, 1, 1), date(2027, 12, 31))
    assert len(sessions) == 242
    assert (sessions[0], sessions[-1]) == (date(2027, 1, 4), date(2027, 12, 30))
    assert calendar.is_trading_day(date(2027, 2, 1))
    closed = [date(2027, 2, 2), date(2027, 3, 1), date(2027, 4, 30), date(2027, 12, 24)]
    assert not [day for day in closed if calendar.is_trading_day(day)]
    assert notes["2027-12-31"] == "117年中華民國開國紀念日"
    assert "2027-02-01" not in notes


def test_days_named_in_a_no_trading_clause_close_without_rows_of_their_own(
    load_fixture: Callable[[str], Any],
) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = [row for row in payload["data"] if row[0] not in ("2027-02-02", "2027-02-03")]
    days = schedule_days({**payload, "data": rows})
    assert days[date(2027, 2, 1)] == (True, "")
    assert days[date(2027, 2, 2)] == (False, SETTLEMENT_ONLY)
    assert days[date(2027, 2, 3)] == (False, SETTLEMENT_ONLY)


def test_a_no_trading_clause_without_a_date_is_about_its_own_row(
    load_fixture: Callable[[str], Any],
) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = _last_day_row("颱風來襲。市場無交易。")
    assert schedule_days({**payload, "data": rows}) == {date(2027, 2, 1): (False, "市場無交易")}


def test_a_no_trading_clause_naming_an_impossible_day_is_a_schema_change(
    load_fixture: Callable[[str], Any],
) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = [["2027-02-01", "農曆春節前最後交易日", "2月30日市場無交易。"]]
    with pytest.raises(SchemaChangedError, match="2月30日"):
        schedule_days({**payload, "data": rows})


def test_a_settlement_only_row_of_its_own_drops_the_full_stop(
    load_fixture: Callable[[str], Any],
) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = [row for row in payload["data"] if row[0] == "2027-02-02"]
    assert rows[0][1].endswith("。")
    assert schedule_days({**payload, "data": rows}) == {date(2027, 2, 2): (False, SETTLEMENT_ONLY)}


def test_the_2023_wording_reads_the_same_way(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture("twse/reference/holiday_schedule_2023.json")
    days = schedule_days(payload)
    assert days[date(2023, 1, 17)] == (True, "")
    assert days[date(2023, 1, 18)] == (False, SETTLEMENT_ONLY)
    assert days[date(2023, 1, 19)] == (False, SETTLEMENT_ONLY)
    assert days[date(2023, 1, 30)] == (True, "")
    calendar, _ = build_calendar((), date(2022, 12, 31), [payload], start=date(2023, 1, 1))
    assert calendar.is_trading_day(date(2023, 1, 17))
    assert len(calendar.between(date(2023, 1, 1), date(2023, 12, 31))) == SCHEDULED_SESSIONS_2023


def test_a_no_trading_clause_naming_a_range_closes_every_day_of_it(
    load_fixture: Callable[[str], Any],
) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = _last_day_row("農曆春節前最後交易。2月2日至2月4日市場無交易，僅辦理結算交割作業。")
    assert schedule_days({**payload, "data": rows}) == {
        date(2027, 2, 1): (True, ""),
        date(2027, 2, 2): (False, SETTLEMENT_ONLY),
        date(2027, 2, 3): (False, SETTLEMENT_ONLY),
        date(2027, 2, 4): (False, SETTLEMENT_ONLY),
    }


def test_weekdays_in_brackets_do_not_stay_in_the_note(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = _last_day_row("2月2日（星期二）及2月3日（星期三）市場無交易，僅辦理結算交割作業。")
    assert schedule_days({**payload, "data": rows}) == {
        date(2027, 2, 1): (True, ""),
        date(2027, 2, 2): (False, SETTLEMENT_ONLY),
        date(2027, 2, 3): (False, SETTLEMENT_ONLY),
    }


def test_a_december_row_naming_a_january_day_means_the_next_year(
    load_fixture: Callable[[str], Any],
) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = [["2027-12-30", "國曆新年前最後交易日", "12月31日至1月3日市場無交易。"]]
    days = schedule_days({**payload, "data": rows})
    assert days[date(2027, 12, 30)] == (True, "")
    assert sorted(day for day, (is_open, _) in days.items() if not is_open) == [
        date(2027, 12, 31),
        date(2028, 1, 1),
        date(2028, 1, 2),
        date(2028, 1, 3),
    ]


def test_a_range_that_runs_backwards_is_a_schema_change(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = _last_day_row("2月4日至2月2日市場無交易。")
    with pytest.raises(SchemaChangedError, match="2月4日至2月2日"):
        schedule_days({**payload, "data": rows})


def test_a_day_named_after_the_no_trading_words_is_not_closed(
    load_fixture: Callable[[str], Any],
) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = _last_day_row("2月2日市場無交易，僅辦理結算交割作業，2月4日恢復交易。")
    assert schedule_days({**payload, "data": rows}) == {
        date(2027, 2, 1): (True, ""),
        date(2027, 2, 2): (False, "市場無交易，僅辦理結算交割作業，2月4日恢復交易"),
    }


def test_a_written_year_places_the_named_day(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = [["2027-06-08", "端午節前最後交易日", "117年1月3日市場無交易。"]]
    assert schedule_days({**payload, "data": rows}) == {
        date(2027, 6, 8): (True, ""),
        date(2028, 1, 3): (False, "市場無交易"),
    }


def test_a_trading_day_another_row_closes_is_a_schema_change(
    load_fixture: Callable[[str], Any],
) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = [
        *_last_day_row("2月2日及2月11日市場無交易。"),
        ["2027-02-11", "農曆春節後開始交易日", "農曆春節後開始交易。"],
    ]
    with pytest.raises(SchemaChangedError, match="2027-02-11 a trading day"):
        schedule_days({**payload, "data": rows})


@pytest.mark.parametrize(
    "detail",
    [
        "2月2、3日市場無交易。",
        "2/2及2/3市場無交易。",
        "2 月 2 日市場無交易。",
        "2月2日至3日市場無交易。",
        "2月2日~2月4日市場無交易。",
        "2月2日起至2月4日止市場無交易。",
        "2027年2月2日市場無交易。",
        "二月二日市場無交易。",
        "2月1日為農曆春節前最後交易日，2月2日及2月3日市場無交易。",
        "2月2日市場無交易，2月3日亦無交易。",
    ],
)
def test_a_no_trading_clause_read_only_in_part_is_a_schema_change(
    load_fixture: Callable[[str], Any], detail: str
) -> None:
    payload = load_fixture(PUBLISHED_2027)
    with pytest.raises(SchemaChangedError, match="cannot be read"):
        schedule_days({**payload, "data": _last_day_row(detail)})


@pytest.mark.parametrize(
    ("detail", "closed"),
    [
        ("2月2日（星期二）至2月4日（星期四）市場無交易，僅辦理結算交割作業。", [2, 3, 4]),
        ("2月2日(星期二)及2月3日(星期三)市場無交易，僅辦理結算交割作業。", [2, 3]),
        ("2月2日、2月3日市場無交易，僅辦理結算交割作業。", [2, 3]),
        ("2月2日與2月3日市場無交易，僅辦理結算交割作業。", [2, 3]),
        ("2月2日和2月3日市場無交易，僅辦理結算交割作業。", [2, 3]),
    ],
)
def test_the_ways_of_joining_named_days_all_read_and_leave_a_clean_note(
    load_fixture: Callable[[str], Any], detail: str, closed: list[int]
) -> None:
    payload = load_fixture(PUBLISHED_2027)
    expected: dict[date, tuple[bool, str]] = {date(2027, 2, 1): (True, "")}
    expected.update({date(2027, 2, of_month): (False, SETTLEMENT_ONLY) for of_month in closed})
    assert schedule_days({**payload, "data": _last_day_row(detail)}) == expected


def test_a_january_row_naming_a_december_day_means_the_year_before(
    load_fixture: Callable[[str], Any],
) -> None:
    payload = load_fixture(PUBLISHED_2027)
    rows = [["2027-01-04", "國曆新年開始交易日", "12月31日市場無交易。"]]
    assert schedule_days({**payload, "data": rows}) == {
        date(2027, 1, 4): (True, ""),
        date(2026, 12, 31): (False, "市場無交易"),
    }


@pytest.mark.parametrize("year", [2021, 2026])
def test_a_recorded_schedule_is_a_whole_year(load_fixture: Callable[[str], Any], year: int) -> None:
    require_whole_year(load_fixture(f"twse/reference/holiday_schedule_{year}.json"))


def test_an_unpublished_schedule_is_not_a_whole_year(load_fixture: Callable[[str], Any]) -> None:
    with pytest.raises(SchemaChangedError, match="lists no day of that year"):
        require_whole_year(load_fixture("twse/reference/holiday_schedule_2027.json"))
