from collections.abc import Callable
from datetime import date, time, timedelta
from typing import Any

import pytest

from apps.etl.models import DataQualityLog
from apps.etl.tasks import US_CLOSE_CHECK
from apps.market.adjust import ADJ_CHECK
from apps.market.calendar import TradingDays
from apps.market.models import TradingCalendar
from apps.research import tasks
from apps.research.models import HoldoutScope
from config.celery import app
from tests.research.test_tasks import (
    Severity,
    calendar_rows,
    fast,
    gate,
    marker,
    now,
    ran,
    record,
    retrains,
    retries_run_in_place,
    sent,
    steps,
)

pytestmark = pytest.mark.django_db
FIXTURES = (calendar_rows, fast, gate, now, ran, retrains, retries_run_in_place, sent)
DAY = date(2026, 11, 25)
RECENT = date(2026, 11, 20)
OLD = date(2026, 8, 3)


@pytest.fixture(autouse=True)
def summer_sessions() -> None:
    """Weekday sessions back to June, so sixty sessions before DAY are on the calendar."""
    first = date(2026, 6, 1)
    days = (first + timedelta(days=n) for n in range((date(2026, 10, 1) - first).days))
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5) for day in days
    )


@pytest.fixture(autouse=True)
def evening_clock(now: Callable[[date, time], None]) -> None:
    now(DAY, time(23, 0))


def changed(detail: dict[str, Any], day: date = DAY) -> DataQualityLog:
    return DataQualityLog.objects.create(
        date=day, check_name=ADJ_CHECK, severity=Severity.INFO, detail=detail
    )


def rebuilds() -> list[DataQualityLog]:
    return list(DataQualityLog.objects.filter(check_name=tasks.REBUILD_CHECK).order_by("pk"))


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def command(name: str, **options: Any) -> str:
        calls.append((name, options))
        return f"{name} ok"

    monkeypatch.setattr(tasks, "_command", command)
    return calls


@pytest.fixture
def two_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(app.conf, "CELERY_WORKER_CONCURRENCY", 2)


def test_a_recent_change_rebuilds_labels_and_features_from_its_earliest_session(
    commands: list[tuple[str, dict[str, Any]]], sent: list[Any]
) -> None:
    first = changed({"stocks": {"2330": "2026-11-23", "2317": "2026-11-24"}})
    second = changed({"earliest": RECENT.isoformat(), "count": 900})
    calendar = TradingDays.from_db()

    summary = tasks.STEPS["rebuild"](DAY)

    assert commands == [
        ("build_labels", {"start": calendar.shift(RECENT, -21), "end": DAY}),
        ("build_features", {"start": RECENT, "end": calendar.shift(DAY, -1)}),
    ]
    assert summary == "build_labels ok; build_features ok"
    [done] = rebuilds()
    assert done.severity == Severity.INFO
    assert done.detail == {
        "earliest": RECENT.isoformat(),
        "stocks": 902,
        "notes": {str(first.pk): first.detail, str(second.pk): second.detail},
        "limit_days": 60,
        "handled": "rebuilt",
    }
    assert sent == []
    assert tasks._open_notes() == []


def test_the_rebuild_never_writes_to_the_data_layer_s_rows(
    commands: list[tuple[str, dict[str, Any]]],
) -> None:
    note = changed({"stocks": {"2330": RECENT.isoformat()}})

    tasks.STEPS["rebuild"](DAY)

    note.refresh_from_db()
    assert note.detail == {"stocks": {"2330": RECENT.isoformat()}}


def test_a_change_on_the_day_itself_needs_only_its_labels(
    commands: list[tuple[str, dict[str, Any]]],
) -> None:
    changed({"stocks": {"2330": DAY.isoformat()}})

    tasks.STEPS["rebuild"](DAY)

    assert commands == [
        ("build_labels", {"start": TradingDays.from_db().shift(DAY, -21), "end": DAY})
    ]
    assert tasks._open_notes() == []


def test_a_change_older_than_sixty_sessions_is_sent_for_a_rebuild_by_hand(
    commands: list[tuple[str, dict[str, Any]]], sent: list[Any]
) -> None:
    named = changed({"stocks": {"2330": OLD.isoformat(), "2317": RECENT.isoformat()}})
    counted = changed({"earliest": "2026-08-10", "count": 900})

    tasks.STEPS["rebuild"](DAY)

    assert commands == []
    [warning] = rebuilds()
    assert warning.severity == Severity.WARNING
    assert warning.detail == {
        "earliest": OLD.isoformat(),
        "stocks": 902,
        "notes": {str(named.pk): named.detail, str(counted.pk): counted.detail},
        "limit_days": 60,
        "handled": "by_hand",
    }
    assert sent == [(DAY, tasks.REBUILD_CHECK, Severity.WARNING)]
    assert tasks._open_notes() == []


def test_an_old_note_goes_by_hand_while_the_recent_ones_are_rebuilt(
    commands: list[tuple[str, dict[str, Any]]], sent: list[Any]
) -> None:
    old = changed({"stocks": {"2330": OLD.isoformat()}})
    recent = changed({"stocks": {"2317": RECENT.isoformat()}})

    tasks.STEPS["rebuild"](DAY)

    assert [options["start"] for name, options in commands if name == "build_features"] == [RECENT]
    warning, done = rebuilds()
    assert (warning.severity, set(warning.detail["notes"])) == (Severity.WARNING, {str(old.pk)})
    assert (done.severity, set(done.detail["notes"])) == (Severity.INFO, {str(recent.pk)})
    assert sent == [(DAY, tasks.REBUILD_CHECK, Severity.WARNING)]
    assert tasks._open_notes() == []


def test_a_change_right_at_the_limit_is_still_rebuilt(
    commands: list[tuple[str, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "RECENT_SESSIONS", 11)
    calendar = TradingDays.from_db()
    edge = calendar.shift(DAY, -10)
    changed({"stocks": {"2330": edge.isoformat()}})

    tasks.STEPS["rebuild"](DAY)

    assert [name for name, _ in commands] == ["build_labels", "build_features"]
    changed({"stocks": {"2330": calendar.shift(edge, -1).isoformat()}})
    commands.clear()

    tasks.STEPS["rebuild"](DAY)

    assert commands == []


def test_the_evening_recomputes_sixty_sessions() -> None:
    assert tasks.RECENT_SESSIONS == 60


def test_the_evening_records_recent_notes_as_covered_and_sends_old_ones(
    commands: list[tuple[str, dict[str, Any]]], sent: list[Any]
) -> None:
    old = changed({"stocks": {"2330": OLD.isoformat()}})
    edge = TradingDays.from_db().shift(DAY, -59)
    recent = changed({"stocks": {"2317": edge.isoformat()}})

    summary = tasks.STEPS["rebuild_covered"](DAY)

    assert commands == []
    warning, covered = rebuilds()
    assert (warning.severity, set(warning.detail["notes"])) == (Severity.WARNING, {str(old.pk)})
    assert covered.severity == Severity.INFO
    assert covered.detail == {
        "earliest": edge.isoformat(),
        "stocks": 1,
        "notes": {str(recent.pk): recent.detail},
        "limit_days": 60,
        "handled": "covered",
    }
    assert summary.endswith(tasks.COVERED)
    assert sent == [(DAY, tasks.REBUILD_CHECK, Severity.WARNING)]
    assert tasks._open_notes() == []


def test_a_change_after_the_day_waits_for_a_later_session(
    commands: list[tuple[str, dict[str, Any]]],
) -> None:
    later = changed({"stocks": {"2330": "2026-11-26"}})
    sooner = changed({"stocks": {"2317": RECENT.isoformat()}})

    tasks.STEPS["rebuild"](DAY)

    assert [options["start"] for _, options in commands][1] == RECENT
    assert tasks._open_notes() == [later]
    assert set(rebuilds()[0].detail["notes"]) == {str(sooner.pk)}


def test_a_note_that_grew_after_its_rebuild_is_rebuilt_again(
    commands: list[tuple[str, dict[str, Any]]],
) -> None:
    note = changed({"stocks": {"2330": RECENT.isoformat()}})
    tasks.STEPS["rebuild"](DAY)
    grown = {"stocks": {"2330": RECENT.isoformat(), "2317": "2026-11-24"}}
    DataQualityLog.objects.filter(pk=note.pk).update(detail=grown)

    assert [open_note.pk for open_note in tasks._open_notes()] == [note.pk]

    tasks.STEPS["rebuild"](DAY)

    assert tasks._open_notes() == []


def test_any_rebuild_that_read_a_note_as_it_reads_now_closes_it() -> None:
    grown = {"stocks": {"2330": RECENT.isoformat(), "2317": "2026-11-24"}}
    note = changed(grown)
    first = {"stocks": {"2330": RECENT.isoformat()}}
    for detail in (grown, first):
        DataQualityLog.objects.create(
            date=DAY,
            check_name=tasks.REBUILD_CHECK,
            severity=Severity.INFO,
            detail={"earliest": RECENT.isoformat(), "stocks": 1, "notes": {str(note.pk): detail}},
        )

    assert tasks._open_notes() == []


def test_notes_past_the_window_are_left_alone(
    commands: list[tuple[str, dict[str, Any]]], now: Callable[[date, time], None]
) -> None:
    changed({"stocks": {"2330": RECENT.isoformat()}})
    now(DAY + timedelta(days=tasks.NOTE_WINDOW + 1), time(23, 0))

    assert tasks._open_notes() == []


def test_a_failed_rebuild_warns_and_leaves_its_notes_open(
    monkeypatch: pytest.MonkeyPatch, sent: list[Any]
) -> None:
    changed({"stocks": {"2330": RECENT.isoformat()}})

    def command(name: str, **options: Any) -> str:
        raise RuntimeError("feature_daily is locked")

    monkeypatch.setattr(tasks, "_command", command)

    summary = tasks.STEPS["rebuild"](DAY)

    assert summary == f"rebuild failed, see {tasks.REBUILD_CHECK}"
    [warning] = rebuilds()
    assert warning.detail == {
        "error": "RuntimeError: feature_daily is locked",
        "limit_days": 60,
        "handled": "failed",
    }
    assert sent == [(DAY, tasks.REBUILD_CHECK, Severity.WARNING)]
    assert len(tasks._open_notes()) == 1


def test_an_unreadable_note_warns_instead_of_failing_the_step(sent: list[Any]) -> None:
    changed({"stocks": {}})

    summary = tasks.STEPS["rebuild"](DAY)

    assert summary == f"rebuild failed, see {tasks.REBUILD_CHECK}"
    [warning] = rebuilds()
    assert warning.detail["error"].startswith("ValueError")
    assert sent == [(DAY, tasks.REBUILD_CHECK, Severity.WARNING)]


def test_a_failed_rebuild_lets_the_morning_predict(
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    now: Callable[[date, time], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed({"stocks": {"2330": RECENT.isoformat()}})
    record(DAY, tasks.EVENING_CHECK)
    record(DAY, US_CLOSE_CHECK)
    now(DAY + timedelta(days=1), time(6, 0))

    def command(name: str, **options: Any) -> str:
        raise RuntimeError("feature_daily is locked")

    monkeypatch.setitem(tasks.STEPS, "rebuild", tasks._rebuild)
    monkeypatch.setattr(tasks, "_command", command)

    tasks.morning.apply().get()

    assert steps(ran) == ["features", "predict", "monitor"]
    assert marker(DAY, tasks.MORNING_CHECK).severity == Severity.INFO


def test_an_open_note_adds_a_last_covered_step_to_the_evening(
    gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    changed({"stocks": {"2330": RECENT.isoformat()}})

    tasks.evening.apply().get()

    assert steps(ran) == ["recent_features", "labels", "rebuild_covered"]


def test_an_open_note_adds_a_last_rebuild_to_the_morning(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    changed({"stocks": {"2330": RECENT.isoformat()}})
    record(DAY, tasks.EVENING_CHECK)
    record(DAY, US_CLOSE_CHECK)
    now(DAY + timedelta(days=1), time(6, 0))

    tasks.morning.apply().get()

    assert steps(ran) == ["features", "predict", "monitor", "rebuild"]


def test_a_morning_that_makes_up_a_missed_evening_recomputes_its_sixty_sessions(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    changed({"stocks": {"2330": RECENT.isoformat()}})
    record(DAY, tasks.EVENING_CHECK, Severity.WARNING)
    record(DAY, US_CLOSE_CHECK)
    now(DAY + timedelta(days=1), time(6, 0))

    tasks.morning.apply().get()

    made_up = ["labels", "features", "predict", "monitor", "recent_features"]
    assert steps(ran) == [*made_up, "rebuild_covered"]


def test_no_open_note_no_rebuild(
    gate: list[bool], ran: list[tuple[str, date, bool]], commands: list[tuple[str, dict[str, Any]]]
) -> None:
    changed({"stocks": {"2330": RECENT.isoformat()}})
    tasks._rebuild(DAY)

    tasks.evening.apply().get()

    assert steps(ran) == ["recent_features", "labels"]


@pytest.mark.usefixtures("two_workers")
def test_the_evening_refuses_to_run_beside_another_worker_process(
    gate: list[bool], ran: list[tuple[str, date, bool]], sent: list[Any]
) -> None:
    tasks.evening.apply().get()

    assert ran == []
    refused = marker(DAY, tasks.EVENING_CHECK)
    assert (refused.severity, refused.detail) == (
        Severity.WARNING,
        {"reason": tasks.CONCURRENT, "concurrency": 2},
    )
    assert sent == [(DAY, tasks.EVENING_CHECK, Severity.WARNING)]


def test_an_unset_worker_count_is_refused_too(
    gate: list[bool], ran: list[tuple[str, date, bool]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(app.conf, "CELERY_WORKER_CONCURRENCY", None)

    tasks.evening.apply().get()

    assert ran == []
    assert marker(DAY, tasks.EVENING_CHECK).detail["concurrency"] is None


@pytest.mark.usefixtures("two_workers")
def test_the_morning_refuses_to_run_beside_another_worker_process(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    record(DAY, US_CLOSE_CHECK)
    now(DAY + timedelta(days=1), time(6, 0))

    tasks.morning.apply().get()

    assert ran == []
    assert marker(DAY, tasks.MORNING_CHECK).detail["reason"] == tasks.CONCURRENT


@pytest.mark.usefixtures("two_workers")
def test_the_retrain_refuses_to_run_beside_another_worker_process(
    now: Callable[[date, time], None], sent: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []
    monkeypatch.setattr(tasks, "build_run", lambda scope, *args: built.append(scope))
    now(date(2026, 11, 3), time(6, 30))

    tasks.retrain.apply(args=("2026-11-02",))

    assert built == []
    refused = marker(date(2026, 11, 2), tasks.RETRAIN_CHECK).detail["refused"]
    assert refused == dict.fromkeys(HoldoutScope.values, "worker concurrency 2, not 1")
    assert sent == [(date(2026, 11, 2), tasks.RETRAIN_CHECK, Severity.WARNING)]
