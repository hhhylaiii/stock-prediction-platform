from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from itertools import count
from pathlib import Path
from typing import Any

import pytest
from django.conf import settings
from django.test import override_settings

from apps.etl.models import DataQualityLog
from apps.etl.tasks import US_CLOSE_CHECK
from apps.research import production, tasks
from apps.research.models import HoldoutScope, LabelDaily, ModelRun
from apps.research.production import DESIGNS, build_run
from apps.research.promotion import PROMOTION_CHECK, Promoted
from apps.research.runs import RunError
from tests.research.run_panels import claim_all
from tests.research.test_tasks import (
    TAIPEI,
    WEDNESDAY,
    Severity,
    calendar_rows,
    fast,
    gate,
    marker,
    now,
    ran,
    record,
    retries_run_in_place,
    sent,
    steps,
)

pytestmark = pytest.mark.django_db
FIXTURES = (calendar_rows, fast, gate, now, ran, retries_run_in_place, sent)
FIRST = date(2026, 11, 2)
FIFTH = date(2026, 11, 6)
SIXTH = date(2026, 11, 9)
FROZEN = {"rank": "rank-wf1", "return": "return-wf1", "interval": None}
RUN = {"elapsed_s": 30.0, "peak_rss_mb": 512.0}


@pytest.fixture(autouse=True)
def memory(monkeypatch: pytest.MonkeyPatch) -> list[float | None]:
    """Free-memory readings in turn, the last one repeating; each build takes 30 s at 512 MB."""
    readings: list[float | None] = [80.0]
    clock = count(step=30.0)
    monkeypatch.setattr(
        tasks, "free_percent", lambda: readings.pop(0) if len(readings) > 1 else readings[0]
    )
    monkeypatch.setattr(tasks, "peak_rss_mb", lambda: 512.0)
    monkeypatch.setattr(tasks, "monotonic", lambda: float(next(clock)))
    return readings


@pytest.fixture(autouse=True)
def retrain_hours(now: Callable[[date, time], None]) -> None:
    """A clock inside the hours a retrain may start in, unless a test sets its own."""
    now(FIRST + timedelta(days=1), time(6, 30))


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    for scope, key in FROZEN.items():
        monkeypatch.setitem(DESIGNS, scope, replace(DESIGNS[scope], research_run_key=key))


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str, date, Path]]:
    calls: list[tuple[str, str, str, date, Path]] = []

    def build(scope: str, run_key: str, research: str, as_of: date, directory: Path) -> Any:
        calls.append((scope, run_key, research, as_of, directory))
        return None

    monkeypatch.setattr(tasks, "build_run", build)
    return calls


@pytest.fixture
def queued(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    days: list[str] = []
    monkeypatch.setattr(tasks.retrain, "delay", lambda day: days.append(day))
    return days


@pytest.fixture
def switched_off() -> Iterator[None]:
    with override_settings(RESEARCH_SCHEDULE_ENABLED=False):
        yield


def labelled(through: date) -> None:
    LabelDaily.objects.create(stock_id="2330", date=through, horizon=5)


def retrained(day: date, built: dict[str, str], severity: str = Severity.INFO) -> None:
    DataQualityLog.objects.create(
        date=day,
        check_name=tasks.RETRAIN_CHECK,
        severity=severity,
        detail={"built": built, "refused": {}, "failed": {}},
    )


def tried(run_key: str) -> None:
    DataQualityLog.objects.create(
        date=WEDNESDAY,
        check_name=PROMOTION_CHECK,
        severity=Severity.WARNING,
        detail={"scope": run_key.split("-")[0], "run_key": run_key, "promoted": False},
    )


def go_live_build(run_key: str, as_of: date, made: date) -> None:
    ModelRun.objects.create(
        run_id=f"{run_key}.lightgbm.h5",
        run_key=run_key,
        model_type="lightgbm",
        objective="rank",
        version="2026-11",
        train_start=date(2023, 1, 2),
        train_end=as_of,
        lib_versions={},
        feature_hash="f",
        params={"production": {"as_of": as_of.isoformat()}},
        created_at=datetime.combine(made, time(10, 0), tzinfo=TAIPEI),
    )


def retrain(day: date) -> None:
    tasks.retrain.apply(args=(day.isoformat(),))


def morning_of(day: date, now: Callable[[date, time], None], at: time = time(6, 0)) -> None:
    record(day, tasks.EVENING_CHECK)
    record(day, US_CLOSE_CHECK)
    now(day + timedelta(days=1), at)


def alerts(sent: list[Any], check: str) -> list[Any]:
    return [call for call in sent if call[1] == check]


def test_a_build_refuses_a_research_run_other_than_the_frozen_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    claim_all(finished=False)
    monkeypatch.setitem(DESIGNS, "rank", replace(DESIGNS["rank"], research_run_key="rank-wf1"))
    fitted: list[str] = []

    def fit(*args: Any, **kwargs: Any) -> list[str]:
        fitted.append("fit")
        return []

    monkeypatch.setattr(production, "_fit", fit)

    with pytest.raises(RunError, match="rank's frozen design takes its references from rank-wf1"):
        build_run("rank", "rank-prod", "rank-wf2", WEDNESDAY, tmp_path)

    assert fitted == []


def test_a_build_from_the_frozen_research_run_passes_the_freeze_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    claim_all(finished=False)
    monkeypatch.setitem(DESIGNS, "rank", replace(DESIGNS["rank"], research_run_key="rank-wf1"))

    with pytest.raises(RunError, match="no model_run of the research run rank-wf1"):
        build_run("rank", "rank-prod", "rank-wf1", WEDNESDAY, tmp_path)


def test_the_retrain_builds_each_frozen_scope_and_refuses_the_rest(
    frozen: None, builds: list[tuple[str, str, str, date, Path]], sent: list[Any]
) -> None:
    labelled(FIRST)

    retrain(FIRST)

    assert builds == [
        ("rank", "rank-2026-11", "rank-wf1", FIRST, settings.FEATURE_PARQUET_DIR),
        ("return", "return-2026-11", "return-wf1", FIRST, settings.FEATURE_PARQUET_DIR),
    ]
    done = marker(FIRST, tasks.RETRAIN_CHECK)
    assert done.severity == Severity.WARNING
    assert done.detail == {
        "built": {"rank": "rank-2026-11", "return": "return-2026-11"},
        "refused": {"interval": tasks.NOT_FROZEN},
        "failed": {},
        "free_percent": 80.0,
        "runs": {"rank": RUN, "return": RUN},
    }
    assert sent == [(FIRST, tasks.RETRAIN_CHECK, Severity.WARNING)]


def test_a_retrain_of_every_scope_is_recorded_as_info_and_not_sent(
    builds: list[tuple[str, str, str, date, Path]],
    sent: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for scope in HoldoutScope.values:
        design = replace(DESIGNS[scope], research_run_key=f"{scope}-wf1")
        monkeypatch.setitem(DESIGNS, scope, design)
    labelled(FIRST)

    retrain(FIRST)

    assert [scope for scope, *_ in builds] == ["rank", "return", "interval"]
    assert marker(FIRST, tasks.RETRAIN_CHECK).severity == Severity.INFO
    assert sent == []


def test_a_failed_build_leaves_the_other_scopes_built(
    frozen: None, monkeypatch: pytest.MonkeyPatch, sent: list[Any]
) -> None:
    labelled(FIRST)

    def build(scope: str, run_key: str, research: str, as_of: date, directory: Path) -> None:
        if scope == "rank":
            raise RunError("rank-wf1 gives no rolling IC threshold at [20] days")

    monkeypatch.setattr(tasks, "build_run", build)

    retrain(FIRST)

    detail = marker(FIRST, tasks.RETRAIN_CHECK).detail
    assert detail["built"] == {"return": "return-2026-11"}
    assert detail["failed"] == {
        "rank": "RunError: rank-wf1 gives no rolling IC threshold at [20] days"
    }
    assert detail["runs"] == {"rank": RUN, "return": RUN}
    assert sent == [(FIRST, tasks.RETRAIN_CHECK, Severity.WARNING)]


def test_the_retrain_waits_for_labels_through_its_day(
    frozen: None, builds: list[tuple[str, str, str, date, Path]]
) -> None:
    labelled(FIRST - timedelta(days=3))

    retrain(FIRST)

    assert builds == []
    detail = marker(FIRST, tasks.RETRAIN_CHECK).detail
    assert set(detail["refused"]) == set(HoldoutScope.values)
    expected = f"label_daily ends on {FIRST - timedelta(days=3)}, before {FIRST}"
    assert detail["refused"]["rank"] == expected
    assert detail["runs"] == {}


def test_a_scope_built_by_hand_this_month_is_not_built_again(
    frozen: None, builds: list[tuple[str, str, str, date, Path]]
) -> None:
    labelled(FIRST)
    friday = FIRST - timedelta(days=3)
    go_live_build("rank-go-live", friday, FIRST)
    go_live_build("return-go-live", friday, friday)

    retrain(FIRST)

    assert [scope for scope, *_ in builds] == ["return"]
    assert marker(FIRST, tasks.RETRAIN_CHECK).detail["refused"] == {
        "rank": tasks.ALREADY_BUILT,
        "interval": tasks.NOT_FROZEN,
    }


def test_a_retrain_cut_short_builds_its_own_keys_again(
    frozen: None, builds: list[tuple[str, str, str, date, Path]]
) -> None:
    labelled(FIRST)
    go_live_build("rank-2026-11-shadow", FIRST, FIRST + timedelta(days=1))
    go_live_build("rank-2026-11", FIRST, FIRST + timedelta(days=1))

    retrain(FIRST)

    assert [scope for scope, *_ in builds] == ["rank", "return"]


@pytest.mark.parametrize("at", [time(5, 59), time(9, 0), time(23, 30)])
def test_a_retrain_outside_its_hours_leaves_the_month_to_the_next_morning(
    frozen: None,
    builds: list[tuple[str, str, str, date, Path]],
    now: Callable[[date, time], None],
    at: time,
) -> None:
    labelled(FIRST)
    now(FIRST + timedelta(days=1), at)

    retrain(FIRST)

    assert builds == []
    assert not DataQualityLog.objects.exists()


def test_a_month_retrains_once_however_often_its_retrain_is_queued(
    frozen: None, builds: list[tuple[str, str, str, date, Path]]
) -> None:
    labelled(FIRST)
    retrained(FIRST, {"rank": "rank-2026-11"}, Severity.WARNING)

    retrain(FIRST + timedelta(days=1))

    assert builds == []
    assert not DataQualityLog.objects.filter(date=FIRST + timedelta(days=1)).exists()


@pytest.mark.usefixtures("switched_off")
def test_the_retrain_does_nothing_while_the_schedule_is_switched_off(
    frozen: None, builds: list[tuple[str, str, str, date, Path]]
) -> None:
    labelled(FIRST)

    retrain(FIRST)

    assert builds == []
    assert not DataQualityLog.objects.exists()


def test_a_retrain_short_of_memory_waits_for_it(
    frozen: None, builds: list[tuple[str, str, str, date, Path]], memory: list[float | None]
) -> None:
    labelled(FIRST)
    memory[:] = [30.0, 45.0, 70.0]

    retrain(FIRST)

    assert [scope for scope, *_ in builds] == ["rank", "return"]
    assert marker(FIRST, tasks.RETRAIN_CHECK).detail["free_percent"] == 70.0


def test_a_retrain_still_short_of_memory_after_its_retries_refuses_every_scope(
    frozen: None,
    builds: list[tuple[str, str, str, date, Path]],
    memory: list[float | None],
    sent: list[Any],
) -> None:
    labelled(FIRST)
    memory[:] = [30.0]

    retrain(FIRST)

    assert builds == []
    done = marker(FIRST, tasks.RETRAIN_CHECK)
    short = "30% of memory free, under 50%"
    assert done.detail["refused"] == dict.fromkeys(HoldoutScope.values, short)
    assert done.detail["free_percent"] == 30.0
    assert sent == [(FIRST, tasks.RETRAIN_CHECK, Severity.WARNING)]


def test_a_retrain_without_a_memory_reading_goes_ahead(
    frozen: None, builds: list[tuple[str, str, str, date, Path]], memory: list[float | None]
) -> None:
    labelled(FIRST)
    memory[:] = [None]

    retrain(FIRST)

    assert [scope for scope, *_ in builds] == ["rank", "return"]
    assert marker(FIRST, tasks.RETRAIN_CHECK).detail["free_percent"] is None


@pytest.mark.parametrize("session", [WEDNESDAY, SIXTH])
def test_a_morning_past_the_first_sessions_of_its_month_warns_the_retrain_was_skipped(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
    sent: list[Any],
    session: date,
) -> None:
    morning_of(session, now)

    tasks.morning.apply().get()

    assert queued == []
    assert marker(session, tasks.MORNING_CHECK).severity == Severity.INFO
    skipped = marker(session, tasks.RETRAIN_CHECK)
    assert skipped.severity == Severity.WARNING
    assert skipped.detail["refused"] == dict.fromkeys(HoldoutScope.values, tasks.SKIPPED)
    assert alerts(sent, tasks.RETRAIN_CHECK) == [(session, tasks.RETRAIN_CHECK, Severity.WARNING)]


@pytest.mark.parametrize("session", [FIRST, FIFTH])
def test_a_morning_among_the_first_sessions_of_its_month_queues_its_retrain(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
    session: date,
) -> None:
    morning_of(session, now)

    tasks.morning.apply().get()

    assert steps(ran) == ["features", "coverage", "predict", "monitor"]
    assert queued == [session.isoformat()]
    assert not DataQualityLog.objects.filter(check_name=tasks.RETRAIN_CHECK).exists()


def test_a_morning_done_after_the_cutoff_leaves_the_retrain_to_the_next_one(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
) -> None:
    morning_of(FIRST, now, time(9, 0))

    tasks.morning.apply().get()

    assert marker(FIRST, tasks.MORNING_CHECK).severity == Severity.INFO
    assert queued == []
    second = FIRST + timedelta(days=1)
    morning_of(second, now, time(8, 59))

    tasks.morning.apply().get()

    assert queued == [second.isoformat()]


@pytest.mark.parametrize(("on", "at"), [(1, time(23, 30)), (2, time(0, 30))])
def test_a_morning_done_late_at_night_queues_no_retrain(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
    on: int,
    at: time,
) -> None:
    morning_of(FIRST, now)
    now(FIRST + timedelta(days=on), at)

    tasks.morning.apply(args=(FIRST.isoformat(),)).get()

    assert marker(FIRST, tasks.MORNING_CHECK).severity == Severity.INFO
    assert queued == []


def test_the_evening_queues_no_retrain(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
) -> None:
    now(FIRST, time(23, 0))

    tasks.evening.apply().get()

    assert marker(FIRST, tasks.EVENING_CHECK).severity == Severity.INFO
    assert queued == []


def test_a_month_that_has_a_retrain_queues_no_other(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
) -> None:
    retrained(FIRST, {}, Severity.WARNING)
    second = FIRST + timedelta(days=1)
    morning_of(second, now)

    tasks.morning.apply().get()

    assert marker(second, tasks.MORNING_CHECK).severity == Severity.INFO
    assert queued == []


def test_last_month_s_retrain_neither_stops_this_month_s_nor_adds_a_promote_step(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
) -> None:
    retrained(WEDNESDAY, {"rank": "rank-2026-10"})
    morning_of(FIRST, now)

    tasks.morning.apply().get()

    assert steps(ran) == ["features", "coverage", "predict", "monitor"]
    assert queued == [FIRST.isoformat()]


def test_a_failed_chain_queues_no_retrain(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    morning_of(FIRST, now)

    def fails(day: date) -> str:
        raise RuntimeError("no features")

    monkeypatch.setitem(tasks.STEPS, "predict", fails)

    tasks.morning.apply()

    assert marker(FIRST, tasks.MORNING_CHECK).detail["failed"] == "predict"
    assert queued == []


def test_the_morning_after_a_retrain_promotes_its_untried_builds_under_the_lock(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
) -> None:
    retrained(FIRST, {"rank": "rank-2026-11", "return": "return-2026-11"})
    tried("return-2026-11")
    day = FIRST + timedelta(days=1)
    morning_of(day, now)

    tasks.morning.apply().get()

    assert ran[-1] == ("promote", day, True)
    assert steps(ran) == ["features", "coverage", "predict", "monitor", "promote"]


def test_no_promote_step_once_every_build_was_tried(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
) -> None:
    retrained(FIRST, {"rank": "rank-2026-11"})
    tried("rank-2026-11")
    morning_of(FIRST + timedelta(days=1), now)

    tasks.morning.apply().get()

    assert steps(ran) == ["features", "coverage", "predict", "monitor"]


def test_the_promote_step_swaps_each_untried_build_from_today_or_its_later_first_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = {"rank": "rank-2026-11", "return": "return-2026-11", "interval": "interval-2026-11"}
    retrained(FIRST, built)
    tried("interval-2026-11")
    day = FIRST + timedelta(days=1)
    firsts = {"rank-2026-11": FIRST, "return-2026-11": day + timedelta(days=1)}
    swaps: list[tuple[str, date, Path]] = []

    def promoted(run_key: str, on: date, directory: Path) -> Promoted:
        swaps.append((run_key, on, directory))
        return Promoted(run_key == "rank-2026-11", [])

    monkeypatch.setattr(tasks, "first_day", firsts.__getitem__)
    monkeypatch.setattr(tasks, "promote", promoted)

    summary = tasks.STEPS["promote"](day)

    assert swaps == [
        ("rank-2026-11", day, settings.FEATURE_PARQUET_DIR),
        ("return-2026-11", day + timedelta(days=1), settings.FEATURE_PARQUET_DIR),
    ]
    assert summary == "rank-2026-11: promoted; return-2026-11: kept the serving runs"
    assert not DataQualityLog.objects.filter(check_name=tasks.PROMOTE_CHECK).exists()


def test_a_promote_that_fails_is_recorded_and_sent_while_the_others_go_on(
    monkeypatch: pytest.MonkeyPatch, sent: list[Any]
) -> None:
    retrained(FIRST, {"rank": "rank-2026-11", "return": "return-2026-11"})
    day = FIRST + timedelta(days=1)
    swaps: list[str] = []
    error = "the serving run of rank 20 日 changed since its gates were checked"

    def promoted(run_key: str, on: date, directory: Path) -> Promoted:
        swaps.append(run_key)
        if run_key == "rank-2026-11":
            raise RunError(error)
        return Promoted(True, [])

    monkeypatch.setattr(tasks, "first_day", lambda run_key: day)
    monkeypatch.setattr(tasks, "promote", promoted)

    summary = tasks.STEPS["promote"](day)

    assert swaps == ["rank-2026-11", "return-2026-11"]
    expected = f"return-2026-11: promoted; rank-2026-11: failed, see {tasks.PROMOTE_CHECK}"
    assert summary == expected
    failure = marker(day, tasks.PROMOTE_CHECK)
    assert failure.severity == Severity.WARNING
    assert failure.detail == {
        "failed": {"rank-2026-11": f"RunError: {error}"},
        "done": ["return-2026-11: promoted"],
    }
    assert sent == [(day, tasks.PROMOTE_CHECK, Severity.WARNING)]


def test_a_failed_promotion_finishes_the_morning_and_is_not_tried_again(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrained(FIRST, {"rank": "rank-2026-11"})
    day = FIRST + timedelta(days=1)
    morning_of(day, now)

    def refused(run_key: str, on: date, directory: Path) -> Promoted:
        raise RunError("the slots of rank-2026-11 already hold rows through 2026-11-04")

    monkeypatch.setitem(tasks.STEPS, "promote", tasks._promote)
    monkeypatch.setattr(tasks, "first_day", lambda run_key: day)
    monkeypatch.setattr(tasks, "promote", refused)

    tasks.morning.apply().get()

    assert marker(day, tasks.MORNING_CHECK).severity == Severity.INFO
    assert set(marker(day, tasks.PROMOTE_CHECK).detail["failed"]) == {"rank-2026-11"}
    assert tasks._pending(day + timedelta(days=1)) == []


def test_a_morning_that_finds_its_last_session_unrun_sends_a_missed_warning(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    sent: list[Any],
    queued: list[str],
) -> None:
    before = WEDNESDAY - timedelta(days=1)
    record(before, tasks.MORNING_CHECK)
    thursday = WEDNESDAY + timedelta(days=1)
    morning_of(thursday, now)

    tasks.morning.apply().get()

    missed = marker(WEDNESDAY, tasks.MORNING_CHECK)
    assert (missed.severity, missed.detail) == (Severity.WARNING, {"reason": tasks.MISSED})
    assert alerts(sent, tasks.MORNING_CHECK) == [(WEDNESDAY, tasks.MORNING_CHECK, Severity.WARNING)]
    assert {day for _, day, _ in ran} == {thursday}


def test_a_monday_session_finds_the_friday_before_it_unrun(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
) -> None:
    friday = FIRST - timedelta(days=3)
    record(friday - timedelta(days=1), tasks.MORNING_CHECK)
    morning_of(FIRST, now)

    tasks.morning.apply().get()

    assert marker(friday, tasks.MORNING_CHECK).detail == {"reason": tasks.MISSED}


@pytest.mark.parametrize(
    ("last_run", "flagged"), [(date(2026, 10, 21), True), (date(2026, 10, 20), False)]
)
def test_a_missed_warning_looks_back_five_sessions_for_the_schedule(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
    last_run: date,
    flagged: bool,
) -> None:
    record(last_run, tasks.MORNING_CHECK)
    morning_of(WEDNESDAY + timedelta(days=1), now)

    tasks.morning.apply().get()

    missed = DataQualityLog.objects.filter(date=WEDNESDAY, check_name=tasks.MORNING_CHECK)
    assert missed.exists() is flagged


def test_no_missed_warning_on_the_first_morning(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    sent: list[Any],
    queued: list[str],
) -> None:
    morning_of(WEDNESDAY + timedelta(days=1), now)

    tasks.morning.apply().get()

    assert alerts(sent, tasks.MORNING_CHECK) == []
    assert not DataQualityLog.objects.filter(date=WEDNESDAY, check_name=tasks.MORNING_CHECK)


@pytest.mark.parametrize("severity", [Severity.INFO, Severity.WARNING])
def test_no_missed_warning_once_the_last_session_has_a_morning_row(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    sent: list[Any],
    queued: list[str],
    severity: str,
) -> None:
    record(WEDNESDAY - timedelta(days=1), tasks.MORNING_CHECK)
    record(WEDNESDAY, tasks.MORNING_CHECK, severity)
    morning_of(WEDNESDAY + timedelta(days=1), now)

    tasks.morning.apply().get()

    assert alerts(sent, tasks.MORNING_CHECK) == []
    kept = marker(WEDNESDAY, tasks.MORNING_CHECK)
    assert (kept.severity, kept.detail) == (severity, {})
