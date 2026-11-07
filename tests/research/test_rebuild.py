from collections.abc import Callable
from datetime import date, time, timedelta
from typing import Any

import pytest

from apps.etl.models import DataQualityLog
from apps.etl.tasks import US_CLOSE_CHECK
from apps.market.adjust import ADJ_CHECK, REBUILT
from apps.market.calendar import TradingDays
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
OLD = date(2026, 10, 2)


def changed(detail: dict[str, Any], day: date = DAY) -> DataQualityLog:
    return DataQualityLog.objects.create(
        date=day, check_name=ADJ_CHECK, severity=Severity.INFO, detail=detail
    )


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
    commands: list[tuple[str, dict[str, Any]]],
) -> None:
    first = changed({"stocks": {"2330": "2026-11-23", "2317": RECENT.isoformat()}})
    second = changed({"earliest": "2026-11-24", "count": 900})
    start = TradingDays.from_db().shift(RECENT, -21)

    summary = tasks.STEPS["rebuild"](DAY)

    assert commands == [
        ("build_labels", {"start": start, "end": DAY}),
        ("build_features", {"start": RECENT, "end": DAY}),
    ]
    assert summary == "build_labels ok; build_features ok"
    for note in (first, second):
        note.refresh_from_db()
        assert note.detail[REBUILT] == DAY.isoformat()


def test_a_change_older_than_thirty_sessions_is_sent_for_a_rebuild_by_hand(
    commands: list[tuple[str, dict[str, Any]]], sent: list[Any]
) -> None:
    note = changed({"stocks": {"2330": OLD.isoformat(), "2317": RECENT.isoformat()}})

    tasks.STEPS["rebuild"](DAY)

    assert commands == []
    warning = marker(DAY, tasks.REBUILD_CHECK)
    assert warning.severity == Severity.WARNING
    assert warning.detail == {"earliest": OLD.isoformat(), "stocks": 2, "limit": 30}
    assert sent == [(DAY, tasks.REBUILD_CHECK, Severity.WARNING)]
    note.refresh_from_db()
    assert note.detail[REBUILT] == DAY.isoformat()


def test_a_change_right_at_the_limit_is_still_rebuilt(
    commands: list[tuple[str, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "REBUILD_LIMIT", 10)
    calendar = TradingDays.from_db()
    edge = calendar.shift(DAY, -10)
    changed({"stocks": {"2330": edge.isoformat()}})

    tasks.STEPS["rebuild"](DAY)

    assert [name for name, _ in commands] == ["build_labels", "build_features"]
    changed({"stocks": {"2330": calendar.shift(edge, -1).isoformat()}})
    commands.clear()

    tasks.STEPS["rebuild"](DAY)

    assert commands == []


def test_the_limit_is_thirty_sessions() -> None:
    assert tasks.REBUILD_LIMIT == 30


def test_a_change_after_the_day_needs_no_rebuild(
    commands: list[tuple[str, dict[str, Any]]],
) -> None:
    note = changed({"stocks": {"2330": "2026-11-26"}})

    tasks.STEPS["rebuild"](DAY)

    assert commands == []
    note.refresh_from_db()
    assert REBUILT in note.detail


def test_a_note_that_grew_while_rebuilding_stays_open_for_the_next_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note = changed({"stocks": {"2330": RECENT.isoformat()}})

    def command(name: str, **options: Any) -> str:
        DataQualityLog.objects.filter(pk=note.pk).update(
            detail={"stocks": {"2330": RECENT.isoformat(), "2317": "2026-11-24"}}
        )
        return "ok"

    monkeypatch.setattr(tasks, "_command", command)

    tasks.STEPS["rebuild"](DAY)

    note.refresh_from_db()
    assert REBUILT not in note.detail


def test_a_failed_rebuild_leaves_its_notes_open(monkeypatch: pytest.MonkeyPatch) -> None:
    note = changed({"stocks": {"2330": RECENT.isoformat()}})

    def command(name: str, **options: Any) -> str:
        raise RuntimeError("feature_daily is locked")

    monkeypatch.setattr(tasks, "_command", command)

    with pytest.raises(RuntimeError, match="feature_daily is locked"):
        tasks.STEPS["rebuild"](DAY)

    note.refresh_from_db()
    assert REBUILT not in note.detail


def test_an_open_note_puts_the_rebuild_first_in_the_evening(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    changed({"stocks": {"2330": RECENT.isoformat()}})
    now(DAY, time(23, 0))

    tasks.evening.apply().get()

    assert steps(ran) == ["rebuild", "features", "labels"]


def test_an_open_note_puts_the_rebuild_first_in_the_morning(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    changed({"stocks": {"2330": RECENT.isoformat()}})
    record(DAY, tasks.EVENING_CHECK)
    record(DAY, US_CLOSE_CHECK)
    now(DAY + timedelta(days=1), time(6, 0))

    tasks.morning.apply().get()

    assert steps(ran) == ["rebuild", "features", "predict", "monitor"]


def test_no_open_note_no_rebuild(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    changed({"stocks": {"2330": RECENT.isoformat()}, REBUILT: "2026-11-21"})
    now(DAY, time(23, 0))

    tasks.evening.apply().get()

    assert steps(ran) == ["features", "labels"]


@pytest.mark.usefixtures("two_workers")
def test_the_evening_refuses_to_run_beside_another_worker_process(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    sent: list[Any],
) -> None:
    now(DAY, time(23, 0))

    tasks.evening.apply().get()

    assert ran == []
    refused = marker(DAY, tasks.EVENING_CHECK)
    assert (refused.severity, refused.detail) == (
        Severity.WARNING,
        {"reason": tasks.CONCURRENT, "concurrency": 2},
    )
    assert sent == [(DAY, tasks.EVENING_CHECK, Severity.WARNING)]


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
