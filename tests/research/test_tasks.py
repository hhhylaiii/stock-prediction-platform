import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from io import StringIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from celery.schedules import crontab
from django.conf import settings
from django.db import connection
from django.test import override_settings

from apps.etl.models import DataQualityLog
from apps.etl.tasks import US_CLOSE_CHECK
from apps.market.calendar import TradingDays
from apps.market.models import TradingCalendar
from apps.research import tasks
from apps.research.lock import LOCK_ID, ResearchBusyError
from apps.research.monitor import MonitorResult
from apps.research.predict import Predicted, Slot
from config.celery import app

pytestmark = pytest.mark.django_db
TAIPEI = ZoneInfo("Asia/Taipei")
WEDNESDAY = date(2026, 10, 28)
FRIDAY = date(2026, 10, 30)
Severity = DataQualityLog.Severity
FAST = {
    "RESEARCH_SCHEDULE_ENABLED": True,
    "ETL_NOT_READY_COUNTDOWN": 0,
    "ETL_NOT_READY_RETRIES": 2,
    "RESEARCH_US_CLOSE_COUNTDOWN": 0,
    "RESEARCH_US_CLOSE_RETRIES": 2,
    "RESEARCH_LOCK_COUNTDOWN": 0,
    "RESEARCH_LOCK_RETRIES": 2,
    "RESEARCH_RETRAIN_COUNTDOWN": 0,
    "RESEARCH_RETRAIN_RETRIES": 2,
}


@pytest.fixture(autouse=True)
def calendar_rows() -> None:
    first = date(2026, 10, 1)
    days = (first + timedelta(days=n) for n in range(61))
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5) for day in days
    )


@pytest.fixture(autouse=True)
def fast() -> Iterator[None]:
    with override_settings(**FAST):
        yield


@pytest.fixture(autouse=True)
def retries_run_in_place() -> Iterator[None]:
    """Eager apply re-runs a retried task at once, unless errors propagate first."""
    before = app.conf.CELERY_TASK_EAGER_PROPAGATES
    app.conf.CELERY_TASK_EAGER_PROPAGATES = False
    yield
    app.conf.CELERY_TASK_EAGER_PROPAGATES = before


@pytest.fixture(autouse=True)
def retrains(monkeypatch: pytest.MonkeyPatch) -> list[date]:
    """The monthly retrain a finished morning may queue or warn about, kept out of these tests."""
    days: list[date] = []
    monkeypatch.setattr(tasks, "_retrain_after", days.append)
    return days


@pytest.fixture
def now(monkeypatch: pytest.MonkeyPatch) -> Callable[[date, time], None]:
    def _set(day: date, at: time) -> None:
        moment = datetime.combine(day, at, tzinfo=TAIPEI)
        monkeypatch.setattr("apps.research.tasks.timezone.localtime", lambda: moment)

    return _set


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Answers of ready_for_research in turn; the last one repeats."""
    answers = [True]

    def ready(day: date) -> bool:
        return answers.pop(0) if len(answers) > 1 else answers[0]

    monkeypatch.setattr(tasks, "ready_for_research", ready)
    return answers


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, date, bool]]:
    """Each step's name, day, and whether this session held the research lock while it ran."""
    calls: list[tuple[str, date, bool]] = []

    def fake(name: str) -> Callable[[date], str]:
        def step(day: date) -> str:
            calls.append((name, day, holds_lock()))
            return f"{name} ok"

        return step

    for name in tasks.STEPS:
        monkeypatch.setitem(tasks.STEPS, name, fake(name))
    return calls


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[date, str, str]]:
    calls: list[tuple[date, str, str]] = []
    monkeypatch.setattr(
        "apps.research.tasks.dispatch",
        lambda rows: calls.extend((row.date, row.check_name, row.severity) for row in rows),
    )
    return calls


def holds_lock() -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND objid = %s "
            "AND granted AND pid = pg_backend_pid()",
            [LOCK_ID],
        )
        (count,) = cursor.fetchone()
    return bool(count)


def record(day: date, check: str, severity: str = Severity.INFO) -> None:
    DataQualityLog.objects.create(date=day, check_name=check, severity=severity, detail={})


def marker(day: date, check: str) -> DataQualityLog:
    return DataQualityLog.objects.get(date=day, check_name=check)


def ok(name: str) -> dict[str, str]:
    return {"step": name, "result": f"{name} ok"}


def steps(ran: list[tuple[str, date, bool]]) -> list[str]:
    return [name for name, _, _ in ran]


def test_the_scheduled_research_does_nothing_while_switched_off(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    now(WEDNESDAY, time(23, 0))
    with override_settings(RESEARCH_SCHEDULE_ENABLED=False):
        assert tasks.evening.apply().get() == "disabled"
        now(WEDNESDAY + timedelta(days=1), time(6, 0))
        assert tasks.morning.apply().get() == "disabled"
    assert ran == []
    assert not DataQualityLog.objects.exists()


def test_the_evening_chain_builds_features_then_labels_under_the_lock(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    sent: list[tuple[date, str, str]],
) -> None:
    now(WEDNESDAY, time(23, 0))

    tasks.evening.apply().get()

    assert ran == [("features", WEDNESDAY, True), ("labels", WEDNESDAY, True)]
    done = marker(WEDNESDAY, tasks.EVENING_CHECK)
    assert done.severity == Severity.INFO
    assert done.detail == {"steps": [ok("features"), ok("labels")]}
    assert sent == []
    assert not holds_lock()


def test_the_evening_waits_for_the_research_gate_and_runs_once_it_opens(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    now(WEDNESDAY, time(23, 0))
    gate[:] = [False, False, True]

    tasks.evening.apply().get()

    assert steps(ran) == ["features", "labels"]
    assert marker(WEDNESDAY, tasks.EVENING_CHECK).severity == Severity.INFO


def test_the_evening_gives_up_with_a_warning_when_the_gate_stays_shut(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    sent: list[tuple[date, str, str]],
) -> None:
    now(WEDNESDAY, time(23, 0))
    gate[:] = [False]

    assert tasks.evening.apply().get() == "not ready"

    assert ran == []
    done = marker(WEDNESDAY, tasks.EVENING_CHECK)
    assert (done.severity, done.detail) == (Severity.WARNING, {"reason": tasks.GATE_SHUT})
    assert sent == [(WEDNESDAY, tasks.EVENING_CHECK, Severity.WARNING)]


def ticking(now: Callable[[date, time], None], start: datetime) -> Callable[[], None]:
    """Move the clock 30 minutes on each call, as the gate's retries would."""
    moments = [start]

    def tick() -> None:
        moments.append(moments[-1] + timedelta(minutes=30))
        now(moments[-1].date(), moments[-1].time())

    return tick


@pytest.mark.parametrize("opens", [True, False])
def test_the_evening_keeps_its_session_when_the_gate_retries_run_past_midnight(
    now: Callable[[date, time], None],
    ran: list[tuple[str, date, bool]],
    sent: list[tuple[date, str, str]],
    monkeypatch: pytest.MonkeyPatch,
    opens: bool,
) -> None:
    now(WEDNESDAY, time(23, 30))
    tick = ticking(now, datetime.combine(WEDNESDAY, time(23, 30), tzinfo=TAIPEI))
    asked: list[date] = []

    def ready(day: date) -> bool:
        asked.append(day)
        tick()
        return opens and len(asked) == 3

    monkeypatch.setattr(tasks, "ready_for_research", ready)

    tasks.evening.apply().get()

    assert asked == [WEDNESDAY] * 3
    if opens:
        assert ran == [("features", WEDNESDAY, True), ("labels", WEDNESDAY, True)]
        assert marker(WEDNESDAY, tasks.EVENING_CHECK).severity == Severity.INFO
    else:
        assert ran == []
        assert marker(WEDNESDAY, tasks.EVENING_CHECK).detail == {"reason": tasks.GATE_SHUT}
        assert sent == [(WEDNESDAY, tasks.EVENING_CHECK, Severity.WARNING)]
    assert not DataQualityLog.objects.filter(date=WEDNESDAY + timedelta(days=1)).exists()


def test_the_morning_keeps_its_session_when_a_late_wake_waits_past_midnight(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thursday = WEDNESDAY + timedelta(days=1)
    record(WEDNESDAY, tasks.EVENING_CHECK)
    now(thursday, time(23, 40))
    tick = ticking(now, datetime.combine(thursday, time(23, 40), tzinfo=TAIPEI))
    looked: list[date] = []

    def landed(day: date) -> bool:
        looked.append(day)
        tick()
        return False

    monkeypatch.setattr(tasks, "_us_close_landed", landed)

    tasks.morning.apply().get()

    assert looked == [WEDNESDAY] * (FAST["RESEARCH_US_CLOSE_RETRIES"] + 1)
    assert {day for _, day, _ in ran} == {WEDNESDAY}
    assert marker(WEDNESDAY, tasks.MORNING_CHECK).detail["us_close"] == tasks.US_CLOSE_MISSING


def retried(task: Any, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The keyword arguments of each retry the task asks for."""
    calls: list[dict[str, Any]] = []
    original = task.retry

    def retry(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(task, "retry", retry)
    return calls


def test_each_wait_retries_on_its_own_countdown_and_count(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits = {
        "ETL_NOT_READY_COUNTDOWN": 11,
        "RESEARCH_US_CLOSE_COUNTDOWN": 22,
        "RESEARCH_LOCK_COUNTDOWN": 33,
    }
    evening, morning, step = (
        retried(task, monkeypatch) for task in (tasks.evening, tasks.morning, tasks.step)
    )
    with override_settings(**waits):
        now(WEDNESDAY, time(23, 0))
        gate[:] = [False]
        tasks.evening.apply().get()
        gate[:] = [True]
        record(WEDNESDAY, tasks.EVENING_CHECK)
        now(WEDNESDAY + timedelta(days=1), time(6, 0))
        monkeypatch.setattr(tasks, "research_lock", busy_lock(2))
        tasks.morning.apply().get()

    day = (WEDNESDAY.isoformat(),)
    assert [(call["countdown"], call["args"]) for call in evening] == [(11, day)] * 2
    assert [(call["countdown"], call["args"]) for call in morning] == [(22, day)] * 2
    assert [call["countdown"] for call in step] == [33] * 2


@pytest.mark.parametrize(
    ("day", "at", "result"),
    [(date(2026, 10, 31), time(23, 0), "closed"), (WEDNESDAY, time(0, 30), "early")],
)
def test_the_evening_skips_closed_days_and_runs_fired_before_its_slot(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    day: date,
    at: time,
    result: str,
) -> None:
    now(day, at)

    assert tasks.evening.apply().get() == result
    assert ran == []


def test_the_evening_does_not_redo_a_day_it_finished(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    now(WEDNESDAY, time(23, 0))
    record(WEDNESDAY, tasks.EVENING_CHECK)

    assert tasks.evening.apply().get() == "done"
    assert ran == []


def test_the_morning_chain_rebuilds_features_then_predicts_then_monitors(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    sent: list[tuple[date, str, str]],
) -> None:
    record(WEDNESDAY, tasks.EVENING_CHECK)
    record(WEDNESDAY, US_CLOSE_CHECK)
    now(WEDNESDAY + timedelta(days=1), time(6, 0))

    tasks.morning.apply().get()

    assert ran == [
        ("features", WEDNESDAY, True),
        ("predict", WEDNESDAY, True),
        ("monitor", WEDNESDAY, True),
    ]
    done = marker(WEDNESDAY, tasks.MORNING_CHECK)
    assert done.severity == Severity.INFO
    assert done.detail["steps"] == [ok("features"), ok("predict"), ok("monitor")]
    assert sent == []


def test_the_morning_first_builds_the_labels_the_evening_did_not(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    record(WEDNESDAY, tasks.EVENING_CHECK, Severity.WARNING)
    record(WEDNESDAY, US_CLOSE_CHECK)
    now(WEDNESDAY + timedelta(days=1), time(6, 0))

    tasks.morning.apply().get()

    assert steps(ran) == ["labels", "features", "predict", "monitor"]


def test_the_morning_after_friday_runs_on_saturday_and_not_again_until_tuesday(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    record(FRIDAY, tasks.EVENING_CHECK)
    record(FRIDAY, US_CLOSE_CHECK)
    for offset in (1, 2, 3):
        now(FRIDAY + timedelta(days=offset), time(6, 0))
        tasks.morning.apply().get()

    assert {day for _, day, _ in ran} == {FRIDAY}
    assert steps(ran) == ["features", "predict", "monitor"]


def test_a_failed_morning_is_redone_the_next_morning_of_the_same_session(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    record(FRIDAY, tasks.EVENING_CHECK)
    record(FRIDAY, US_CLOSE_CHECK)
    record(FRIDAY, tasks.MORNING_CHECK, Severity.WARNING)
    now(FRIDAY + timedelta(days=2), time(6, 0))

    tasks.morning.apply().get()

    assert steps(ran) == ["features", "predict", "monitor"]
    assert marker(FRIDAY, tasks.MORNING_CHECK).severity == Severity.INFO


def test_the_morning_waits_for_the_us_close_and_then_runs_without_it_with_a_warning(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    sent: list[tuple[date, str, str]],
) -> None:
    record(WEDNESDAY, tasks.EVENING_CHECK)
    now(WEDNESDAY + timedelta(days=1), time(6, 0))

    tasks.morning.apply().get()

    assert steps(ran) == ["features", "predict", "monitor"]
    done = marker(WEDNESDAY, tasks.MORNING_CHECK)
    assert done.severity == Severity.WARNING
    assert done.detail["us_close"] == tasks.US_CLOSE_MISSING
    assert sent == [(WEDNESDAY, tasks.MORNING_CHECK, Severity.WARNING)]


def test_the_morning_picks_up_a_us_close_written_while_it_waited(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record(WEDNESDAY, tasks.EVENING_CHECK)
    now(WEDNESDAY + timedelta(days=1), time(6, 0))
    waits: list[date] = []
    landed = tasks._us_close_landed

    def lands_on_second_look(day: date) -> bool:
        waits.append(day)
        if len(waits) == 2:
            record(day, US_CLOSE_CHECK)
        return landed(day)

    monkeypatch.setattr(tasks, "_us_close_landed", lands_on_second_look)

    tasks.morning.apply().get()

    assert waits == [WEDNESDAY, WEDNESDAY]
    assert marker(WEDNESDAY, tasks.MORNING_CHECK).severity == Severity.INFO


def test_the_morning_does_not_predict_a_day_whose_research_gate_is_shut(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    sent: list[tuple[date, str, str]],
) -> None:
    record(WEDNESDAY, US_CLOSE_CHECK)
    gate[:] = [False]
    now(WEDNESDAY + timedelta(days=1), time(6, 0))

    assert tasks.morning.apply().get() == "gate shut"

    assert ran == []
    done = marker(WEDNESDAY, tasks.MORNING_CHECK)
    assert (done.severity, done.detail) == (Severity.WARNING, {"reason": tasks.GATE_SHUT})
    assert sent == [(WEDNESDAY, tasks.MORNING_CHECK, Severity.WARNING)]


def test_a_failed_step_stops_the_chain_and_records_what_was_done(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    sent: list[tuple[date, str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record(WEDNESDAY, tasks.EVENING_CHECK)
    record(WEDNESDAY, US_CLOSE_CHECK)
    now(WEDNESDAY + timedelta(days=1), time(6, 0))

    def fails(day: date) -> str:
        raise RuntimeError("no serving run for rank 20")

    monkeypatch.setitem(tasks.STEPS, "predict", fails)

    tasks.morning.apply()

    assert steps(ran) == ["features"]
    done = marker(WEDNESDAY, tasks.MORNING_CHECK)
    assert done.severity == Severity.WARNING
    assert done.detail == {
        "failed": "predict",
        "error": "RuntimeError: no serving run for rank 20",
        "steps": [ok("features")],
    }
    assert sent == [(WEDNESDAY, tasks.MORNING_CHECK, Severity.WARNING)]
    assert not holds_lock()


def busy_lock(times: int, only: str = "") -> Callable[[str], Any]:
    """A research lock held elsewhere for the first tries of the steps whose name starts so."""
    tries = {"left": times}
    real = tasks.research_lock

    @contextmanager
    def lock(holder: str) -> Iterator[None]:
        if holder.startswith(only) and tries["left"] > 0:
            tries["left"] -= 1
            raise ResearchBusyError(f"the research lock is held by 'promote {holder}'")
        with real(holder):
            yield

    return lock


def test_a_step_that_finds_the_lock_busy_retries_until_it_is_free(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record(WEDNESDAY, tasks.EVENING_CHECK)
    record(WEDNESDAY, US_CLOSE_CHECK)
    now(WEDNESDAY + timedelta(days=1), time(6, 0))
    monkeypatch.setattr(tasks, "research_lock", busy_lock(2))

    tasks.morning.apply().get()

    assert steps(ran) == ["features", "predict", "monitor"]
    assert marker(WEDNESDAY, tasks.MORNING_CHECK).severity == Severity.INFO


def test_a_later_step_that_waits_for_the_lock_keeps_the_steps_done_before_it(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record(WEDNESDAY, tasks.EVENING_CHECK)
    record(WEDNESDAY, US_CLOSE_CHECK)
    now(WEDNESDAY + timedelta(days=1), time(6, 0))
    monkeypatch.setattr(tasks, "research_lock", busy_lock(1, only="predict"))

    tasks.morning.apply().get()

    assert steps(ran) == ["features", "predict", "monitor"]
    done = marker(WEDNESDAY, tasks.MORNING_CHECK)
    assert done.detail["steps"] == [ok("features"), ok("predict"), ok("monitor")]


def test_a_step_that_never_gets_the_lock_fails_the_chain(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    record(WEDNESDAY, tasks.EVENING_CHECK)
    record(WEDNESDAY, US_CLOSE_CHECK)
    now(WEDNESDAY + timedelta(days=1), time(6, 0))
    monkeypatch.setattr(tasks, "research_lock", busy_lock(3))

    with caplog.at_level(logging.ERROR, logger="apps.research.tasks"):
        tasks.morning.apply()

    assert ran == []
    done = marker(WEDNESDAY, tasks.MORNING_CHECK)
    assert done.detail["failed"] == "features"
    assert done.detail["error"].startswith("ResearchBusyError: the research lock is held by")
    assert "research features" in caplog.text


@pytest.mark.parametrize("at", [time(5, 59), time(3, 0)])
def test_the_morning_skips_a_run_fired_before_its_slot(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    at: time,
) -> None:
    now(WEDNESDAY + timedelta(days=1), at)

    assert tasks.morning.apply().get() == "early"
    assert ran == []


def test_a_marker_is_replaced_by_a_rerun_and_never_raised_to_error(
    now: Callable[[date, time], None], gate: list[bool], ran: list[tuple[str, date, bool]]
) -> None:
    record(WEDNESDAY, tasks.EVENING_CHECK, Severity.WARNING)
    now(WEDNESDAY, time(23, 0))

    tasks.evening.apply().get()

    rows = DataQualityLog.objects.filter(date=WEDNESDAY, check_name=tasks.EVENING_CHECK)
    assert [row.severity for row in rows] == [Severity.INFO]
    assert not DataQualityLog.objects.filter(severity=Severity.ERROR).exists()


def test_the_beat_schedule_fires_both_chains_at_their_configured_times() -> None:
    schedule = settings.CELERY_BEAT_SCHEDULE
    assert schedule["research-evening"] == {
        "task": "research.evening",
        "schedule": crontab(hour=23, minute=0),
    }
    assert schedule["research-morning"] == {
        "task": "research.morning",
        "schedule": crontab(hour=6, minute=0),
    }


def test_the_real_steps_run_the_research_commands_and_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[tuple[str, Any]] = []

    def command(name: str, stdout: StringIO, **options: Any) -> None:
        called.append((name, options))
        stdout.write(f"{name}: 7 rows\n")

    def predicted(day: date, directory: Path) -> Predicted:
        called.append(("predict", (day, directory)))
        return Predicted(written={"r1": 3}, missing=[Slot("quantile", 20)])

    def monitored(day: date, calendar: TradingDays) -> MonitorResult:
        called.append(("monitor", (day, calendar.is_trading_day(day))))
        return MonitorResult(scored=4, logs=(), degraded=("r2",))

    monkeypatch.setattr(tasks, "call_command", command)
    monkeypatch.setattr(tasks, "predict", predicted)
    monkeypatch.setattr(tasks, "run_monitor", monitored)

    assert tasks.STEPS["features"](WEDNESDAY) == "build_features: 7 rows"
    assert tasks.STEPS["labels"](WEDNESDAY) == "build_labels: 7 rows"
    assert tasks.STEPS["predict"](WEDNESDAY) == "r1: 3 rows; no serving run for quantile 20 日"
    assert tasks.STEPS["monitor"](WEDNESDAY) == "4 model_monitor rows, degraded r2"
    assert called == [
        ("build_features", {"date": WEDNESDAY}),
        ("build_labels", {"end": WEDNESDAY}),
        ("predict", (WEDNESDAY, settings.FEATURE_PARQUET_DIR)),
        ("monitor", (WEDNESDAY, True)),
    ]
