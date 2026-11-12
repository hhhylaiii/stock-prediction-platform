from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pytest
import responses
from responses import matchers

from apps.etl import calendar_upkeep, tasks
from apps.etl.calendar_upkeep import (
    CHANGED,
    EXTEND_DATASET,
    REACH,
    Extension,
    extend,
    keep_up,
    reach_level,
    record_changes,
)
from apps.etl.errors import SchemaChangedError
from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.quality import CHECKS, should_run_prediction
from apps.etl.tasks import AUDITED
from apps.market.models import TradingCalendar

SCHEDULE_URL = "https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule"
SCHEDULE_2026 = "twse/reference/holiday_schedule_2026.json"
SCHEDULE_2027 = "twse/reference/holiday_schedule_2027.json"
SESSIONS_2026 = 243

pytestmark = [pytest.mark.django_db, pytest.mark.usefixtures("quiet")]


@pytest.fixture
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[list[tuple[date, str, str]]]:
    calls: list[list[tuple[date, str, str]]] = []
    monkeypatch.setattr(
        "apps.etl.calendar_upkeep.announce",
        lambda rows: calls.append([(row.date, row.check_name, row.severity) for row in rows]),
    )
    return calls


def _calendar(
    start: date,
    end: date,
    provisional_from: date | None = None,
    closed: frozenset[date] = frozenset(),
) -> None:
    """Weekdays trade, except the closed ones; days from provisional_from on are provisional."""
    days = [start + timedelta(days=n) for n in range((end - start).days + 1)]
    TradingCalendar.objects.bulk_create(
        TradingCalendar(
            date=day,
            is_trading_day=day.weekday() < 5 and day not in closed,
            is_provisional=provisional_from is not None and day >= provisional_from,
        )
        for day in days
    )


def _serve(year: int, payload: dict[str, Any]) -> None:
    responses.get(
        SCHEDULE_URL,
        json=payload,
        match=[matchers.query_param_matcher({"date": f"{year}0101", "response": "json"})],
    )


def _stored() -> dict[date, tuple[bool, str, bool]]:
    rows = TradingCalendar.objects.values_list("date", "is_trading_day", "note", "is_provisional")
    return {day: (is_open, note, provisional) for day, is_open, note, provisional in rows}


def _sessions(year: int) -> int:
    return TradingCalendar.objects.filter(date__year=year, is_trading_day=True).count()


def _reach(day: date) -> DataQualityLog:
    return DataQualityLog.objects.get(date=day, check_name=REACH)


@responses.activate
def test_a_published_year_is_appended_without_touching_loaded_days(
    load_fixture: Callable[[str], Any],
) -> None:
    _calendar(date(2025, 10, 1), date(2025, 12, 31), closed=frozenset({date(2025, 12, 24)}))
    TradingCalendar.objects.filter(date=date(2025, 12, 24)).update(note="臨時休市")
    before = _stored()
    _serve(2026, load_fixture(SCHEDULE_2026))
    assert extend(date(2025, 12, 1)) == Extension.EXTENDED
    after = _stored()
    assert {day: row for day, row in after.items() if day.year == 2025} == before
    assert max(after) == date(2026, 12, 31)
    assert len([day for day in after if day.year == 2026]) == 365
    assert _sessions(2026) == SESSIONS_2026
    assert after[date(2026, 1, 1)] == (False, "中華民國開國紀念日", False)
    assert after[date(2026, 1, 2)] == (True, "", False)
    assert not any(provisional for _, _, provisional in after.values())
    run = EtlRun.objects.get(dataset=EXTEND_DATASET)
    assert (run.source, run.status, run.rows) == ("trading_calendar", "success", 365)
    assert (run.start_date, run.end_date) == (date(2026, 1, 1), date(2026, 12, 31))
    assert not DataQualityLog.objects.filter(check_name=CHANGED).exists()


@responses.activate
def test_an_unpublished_year_writes_nothing(load_fixture: Callable[[str], Any]) -> None:
    _calendar(date(2026, 10, 1), date(2026, 12, 31))
    before = _stored()
    _serve(2027, load_fixture(SCHEDULE_2027))
    assert extend(date(2026, 11, 12)) == Extension.UNPUBLISHED
    assert _stored() == before
    assert not EtlRun.objects.exists()


@responses.activate
@pytest.mark.parametrize(
    ("rows", "message"),
    [(slice(0, 1), "260 sessions"), (slice(1, None), "trades on 2026-01-01")],
)
def test_a_schedule_that_cannot_be_a_whole_year_is_refused(
    load_fixture: Callable[[str], Any], rows: slice, message: str
) -> None:
    _calendar(date(2025, 10, 1), date(2025, 12, 31))
    before = _stored()
    payload = load_fixture(SCHEDULE_2026)
    _serve(2026, {**payload, "data": payload["data"][rows]})
    with pytest.raises(SchemaChangedError, match=message):
        extend(date(2025, 12, 1))
    assert _stored() == before
    run = EtlRun.objects.get(dataset=EXTEND_DATASET)
    assert run.status == "failed"
    assert message in run.error


@responses.activate
def test_nothing_is_asked_while_enough_sessions_remain() -> None:
    _calendar(date(2025, 6, 1), date(2025, 12, 31))
    assert extend(date(2025, 6, 2)) == Extension.NOT_NEEDED
    assert len(responses.calls) == 0


@responses.activate
def test_a_calendar_ending_inside_a_year_is_left_alone() -> None:
    _calendar(date(2026, 10, 1), date(2026, 10, 31))
    assert extend(date(2026, 10, 28)) == Extension.NO_YEAR_END
    assert len(responses.calls) == 0


@responses.activate
def test_the_threshold_comes_from_settings(settings: Any) -> None:
    settings.CALENDAR_EXTEND_BELOW = 10
    _calendar(date(2026, 10, 1), date(2026, 12, 31))
    assert extend(date(2026, 11, 12)) == Extension.NOT_NEEDED
    assert len(responses.calls) == 0


@responses.activate
def test_a_published_year_replaces_provisional_days_and_records_the_flips(
    load_fixture: Callable[[str], Any], sent: list[list[tuple[date, str, str]]]
) -> None:
    _calendar(date(2025, 10, 1), date(2026, 12, 31), provisional_from=date(2026, 1, 1))
    _serve(2026, load_fixture(SCHEDULE_2026))
    today = date(2025, 12, 15)
    assert extend(today) == Extension.EXTENDED
    after = _stored()
    assert not any(provisional for _, _, provisional in after.values())
    assert _sessions(2026) == SESSIONS_2026
    row = DataQualityLog.objects.get(check_name=CHANGED)
    assert (row.date, row.severity) == (today, "warning")
    assert row.detail["by"] == ["extend"]
    assert row.detail["days"]["2026-01-01"] == {"was": True, "now": False}
    assert len(row.detail["days"]) == 18
    assert row.detail["earliest"] == "2026-01-01"
    assert sent == [[(today, CHANGED, "warning")]]


@responses.activate
def test_provisional_days_already_passed_keep_their_flag_of_trading(
    load_fixture: Callable[[str], Any],
) -> None:
    _calendar(date(2025, 10, 1), date(2026, 12, 31), provisional_from=date(2026, 1, 1))
    _serve(2026, load_fixture(SCHEDULE_2026))
    assert extend(date(2026, 2, 16)) == Extension.EXTENDED
    after = _stored()
    assert after[date(2026, 1, 1)] == (True, "", False)
    assert after[date(2026, 2, 13)] == (True, "", False)
    assert after[date(2026, 2, 16)][0] is False
    assert after[date(2026, 2, 17)][0] is False
    row = DataQualityLog.objects.get(check_name=CHANGED)
    assert min(row.detail["days"]) == "2026-02-16"


@responses.activate
def test_a_provisional_year_is_asked_for_even_with_many_days_ahead(
    load_fixture: Callable[[str], Any],
) -> None:
    _calendar(date(2026, 10, 1), date(2027, 12, 31), provisional_from=date(2027, 1, 1))
    before = _stored()
    _serve(2027, load_fixture(SCHEDULE_2027))
    assert extend(date(2026, 11, 12)) == Extension.UNPUBLISHED
    assert len(responses.calls) == 1
    assert _stored() == before


@pytest.mark.parametrize(
    ("left", "level"),
    [(200, 0), (61, 0), (60, 1), (41, 1), (40, 2), (26, 2), (25, 3), (0, 3)],
)
def test_the_reach_level_steps_at_sixty_forty_and_twenty_five(left: int, level: int) -> None:
    assert reach_level(left) == level


@responses.activate
def test_the_reach_row_counts_confirmed_sessions_ahead(
    load_fixture: Callable[[str], Any], sent: list[list[tuple[date, str, str]]]
) -> None:
    _calendar(date(2026, 10, 1), date(2027, 12, 31), provisional_from=date(2027, 1, 1))
    _serve(2027, load_fixture(SCHEDULE_2027))
    today = date(2026, 11, 12)
    assert keep_up(today) == Extension.UNPUBLISHED
    row = _reach(today)
    assert row.severity == "warning"
    assert row.detail == {
        "sessions_left": 35,
        "ends": "2026-12-31",
        "provisional_to": "2027-12-31",
        "level": 2,
        "extension": "unpublished",
    }
    assert sent == [[(today, REACH, "warning")]]


@responses.activate
def test_a_level_is_announced_once_until_the_last_one_which_repeats_daily(
    load_fixture: Callable[[str], Any], sent: list[list[tuple[date, str, str]]]
) -> None:
    _calendar(date(2026, 10, 1), date(2026, 12, 31))
    _serve(2027, load_fixture(SCHEDULE_2027))
    for day in (date(2026, 11, 12), date(2026, 11, 13), date(2026, 11, 13)):
        keep_up(day)
    assert sent == [[(date(2026, 11, 12), REACH, "warning")]]
    assert DataQualityLog.objects.filter(check_name=REACH).count() == 2
    for day in (date(2026, 11, 26), date(2026, 11, 27), date(2026, 11, 27)):
        keep_up(day)
    assert _reach(date(2026, 11, 26)).detail["sessions_left"] == 25
    assert [call[0][0] for call in sent[1:]] == [date(2026, 11, 26), date(2026, 11, 27)]


@responses.activate
def test_reaching_far_again_is_announced_as_info(
    load_fixture: Callable[[str], Any], sent: list[list[tuple[date, str, str]]]
) -> None:
    _calendar(date(2025, 10, 1), date(2025, 12, 31))
    _serve(2026, load_fixture(SCHEDULE_2027) | {"queryYear": 2026})
    keep_up(date(2025, 12, 1))
    responses.reset()
    _serve(2026, load_fixture(SCHEDULE_2026))
    today = date(2025, 12, 2)
    assert keep_up(today) == Extension.EXTENDED
    row = _reach(today)
    assert row.severity == "info"
    assert row.detail == {
        "sessions_left": 21 + SESSIONS_2026,
        "ends": "2026-12-31",
        "provisional_to": None,
        "level": 0,
        "extension": "extended",
    }
    assert sent == [[(date(2025, 12, 1), REACH, "warning")], [(today, REACH, "info")]]


def test_a_long_calendar_is_recorded_quietly(sent: list[list[tuple[date, str, str]]]) -> None:
    _calendar(date(2025, 6, 1), date(2025, 12, 31))
    today = date(2025, 6, 2)
    assert keep_up(today) == Extension.NOT_NEEDED
    assert (_reach(today).severity, _reach(today).detail["level"]) == ("info", 0)
    assert sent == []


@responses.activate
def test_a_refused_schedule_is_announced_whatever_the_level(
    load_fixture: Callable[[str], Any], sent: list[list[tuple[date, str, str]]]
) -> None:
    _calendar(date(2026, 8, 1), date(2026, 12, 31))
    payload = load_fixture(SCHEDULE_2026) | {"queryYear": 2027}
    today = date(2026, 10, 1)
    for _ in range(2):
        _serve(2027, {**payload, "data": []})
        keep_up(today)
        responses.reset()
    assert (_reach(today).severity, _reach(today).detail["level"]) == ("info", 0)
    assert sent == []
    for _ in range(2):
        _serve(2027, {**payload, "data": [["2027-01-01", *payload["data"][0][1:]]]})
        assert keep_up(today) == Extension.REFUSED
        responses.reset()
    assert (_reach(today).severity, _reach(today).detail["extension"]) == ("warning", "refused")
    assert sent == [[(today, REACH, "warning")]]
    assert set(EtlRun.objects.values_list("dataset", "status")) == {(EXTEND_DATASET, "failed")}


@responses.activate
def test_a_source_failure_still_records_the_reach(
    sent: list[list[tuple[date, str, str]]],
) -> None:
    _calendar(date(2026, 10, 1), date(2026, 12, 31))
    responses.get(SCHEDULE_URL, status=403)
    today = date(2026, 11, 12)
    assert keep_up(today) == Extension.FAILED
    assert _reach(today).detail["extension"] == "failed"
    assert not EtlRun.objects.exists()


@responses.activate
def test_a_calendar_that_ran_out_is_still_recorded_and_announced(
    load_fixture: Callable[[str], Any], sent: list[list[tuple[date, str, str]]]
) -> None:
    _calendar(date(2026, 10, 1), date(2026, 12, 31))
    for day in (date(2027, 1, 1), date(2027, 1, 2)):
        _serve(2027, load_fixture(SCHEDULE_2027))
        assert keep_up(day) == Extension.UNPUBLISHED
        responses.reset()
    row = _reach(date(2027, 1, 2))
    assert (row.severity, row.detail["sessions_left"], row.detail["level"]) == ("warning", 0, 3)
    assert [call[0][0] for call in sent] == [date(2027, 1, 1), date(2027, 1, 2)]


def test_an_empty_calendar_has_nothing_to_keep_up() -> None:
    assert keep_up(date(2026, 11, 12)) is None
    assert not DataQualityLog.objects.exists()


@responses.activate
def test_a_closed_day_inside_the_calendar_is_skipped() -> None:
    _calendar(date(2026, 10, 1), date(2026, 12, 31))
    assert keep_up(date(2026, 11, 14)) is None
    assert not DataQualityLog.objects.exists()
    assert len(responses.calls) == 0


def test_flips_of_one_day_merge_into_one_row(sent: list[list[tuple[date, str, str]]]) -> None:
    today = date(2026, 11, 12)
    record_changes("refresh_calendar", today, {date(2026, 11, 10): (True, False)})
    record_changes("load_calendar", today, {date(2026, 12, 25): (False, True)})
    record_changes("load_calendar", today, {})
    row = DataQualityLog.objects.get(check_name=CHANGED)
    assert row.detail == {
        "by": ["load_calendar", "refresh_calendar"],
        "days": {
            "2026-11-10": {"was": True, "now": False},
            "2026-12-25": {"was": False, "now": True},
        },
        "earliest": "2026-11-10",
    }
    assert len(sent) == 2


def test_a_day_flipped_back_the_same_day_leaves_the_row(
    sent: list[list[tuple[date, str, str]]],
) -> None:
    today = date(2026, 11, 12)
    record_changes("load_calendar", today, {date(2026, 11, 10): (True, False)})
    record_changes("load_calendar", today, {date(2026, 11, 10): (False, True)})
    row = DataQualityLog.objects.get(check_name=CHANGED)
    assert row.detail["days"] == {"2026-11-10": {"was": True, "now": True}}


def test_the_module_names_its_checks_outside_the_gate() -> None:
    assert not {REACH, CHANGED} & {*CHECKS, *AUDITED}
    assert calendar_upkeep.REACH_BOUNDS == (60, 40, 25)


@responses.activate
@pytest.mark.parametrize("extension", [Extension.UNPUBLISHED, Extension.REFUSED])
def test_the_upkeep_never_moves_the_research_gate(
    load_fixture: Callable[[str], Any], extension: Extension
) -> None:
    _calendar(date(2026, 10, 1), date(2026, 12, 31))
    day = date(2026, 11, 12)
    for source, dataset in tasks._daily_jobs():
        EtlRun.objects.create(
            source=source, dataset=dataset, start_date=day, end_date=day, status="success"
        )
    for check in CHECKS:
        DataQualityLog.objects.create(date=day, check_name=check, severity="info", detail={})

    def gate() -> tuple[str, bool, bool, bool]:
        return (
            tasks._landed(day).severity,
            tasks.etl_complete(day),
            should_run_prediction(day),
            tasks.ready_for_research(day),
        )

    before = gate()
    payload = load_fixture(SCHEDULE_2026) | {"queryYear": 2027}
    rows = [] if extension == Extension.UNPUBLISHED else [["2027-01-01", *payload["data"][0][1:]]]
    _serve(2027, {**payload, "data": rows})
    assert keep_up(day) == extension
    assert _reach(day).severity == "warning"
    assert gate() == before == ("info", True, True, True)
