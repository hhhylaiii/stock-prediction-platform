from collections.abc import Callable
from dataclasses import replace
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

import pytest
from django.conf import settings

from apps.etl.models import DataQualityLog
from apps.etl.tasks import US_CLOSE_CHECK
from apps.research import production, tasks
from apps.research.models import HoldoutScope, LabelDaily
from apps.research.production import DESIGNS, build_run
from apps.research.promotion import PROMOTION_CHECK, Promoted
from apps.research.runs import RunError
from tests.research.run_panels import claim_all
from tests.research.test_tasks import (
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
FROZEN = {"rank": "rank-wf1", "return": "return-wf1", "interval": None}


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


def test_no_scope_names_its_research_run_before_the_design_is_frozen() -> None:
    frozen = {scope: design.research_run_key for scope, design in DESIGNS.items()}
    assert frozen == dict.fromkeys(HoldoutScope.values)


def test_a_build_refuses_a_research_run_other_than_the_frozen_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    claim_all()
    monkeypatch.setitem(DESIGNS, "rank", replace(DESIGNS["rank"], research_run_key="rank-wf1"))
    fitted: list[str] = []

    def fit(*args: Any, **kwargs: Any) -> list[str]:
        fitted.append("fit")
        return []

    monkeypatch.setattr(production, "_fit", fit)

    with pytest.raises(RunError, match="rank's frozen design takes its references from rank-wf1"):
        build_run("rank", "rank-prod", "rank-wf2", WEDNESDAY, tmp_path)

    assert fitted == []


def test_the_retrain_builds_each_frozen_scope_and_refuses_the_rest(
    frozen: None, builds: list[tuple[str, str, str, date, Path]], sent: list[Any]
) -> None:
    labelled(FIRST)

    tasks.retrain.apply(args=(FIRST.isoformat(),)).get()

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

    tasks.retrain.apply(args=(FIRST.isoformat(),)).get()

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

    tasks.retrain.apply(args=(FIRST.isoformat(),)).get()

    detail = marker(FIRST, tasks.RETRAIN_CHECK).detail
    assert detail["built"] == {"return": "return-2026-11"}
    assert detail["failed"] == {
        "rank": "RunError: rank-wf1 gives no rolling IC threshold at [20] days"
    }
    assert sent == [(FIRST, tasks.RETRAIN_CHECK, Severity.WARNING)]


def test_the_retrain_waits_for_labels_through_its_day(
    frozen: None, builds: list[tuple[str, str, str, date, Path]]
) -> None:
    labelled(FIRST - timedelta(days=3))

    tasks.retrain.apply(args=(FIRST.isoformat(),)).get()

    assert builds == []
    refused = marker(FIRST, tasks.RETRAIN_CHECK).detail["refused"]
    assert set(refused) == set(HoldoutScope.values)
    assert refused["rank"] == f"label_daily ends on {FIRST - timedelta(days=3)}, before {FIRST}"


def morning_of(day: date, now: Callable[[date, time], None]) -> None:
    record(day, tasks.EVENING_CHECK)
    record(day, US_CLOSE_CHECK)
    now(day + timedelta(days=1), time(6, 0))


def test_a_morning_past_the_first_sessions_of_its_month_queues_no_retrain(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
) -> None:
    morning_of(WEDNESDAY, now)

    tasks.morning.apply().get()

    assert queued == []


def test_the_first_morning_of_a_month_queues_its_retrain_once_the_chain_is_done(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
) -> None:
    morning_of(FIRST, now)

    tasks.morning.apply().get()

    assert steps(ran) == ["features", "predict", "monitor"]
    assert queued == [FIRST.isoformat()]


def test_a_month_that_has_a_retrain_queues_no_other(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    queued: list[str],
) -> None:
    retrained(FIRST, {}, Severity.WARNING)
    morning_of(FIRST + timedelta(days=1), now)

    tasks.morning.apply().get()

    assert queued == []


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
    assert steps(ran) == ["features", "predict", "monitor", "promote"]


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

    assert steps(ran) == ["features", "predict", "monitor"]


def test_the_promote_step_swaps_each_untried_build_from_its_first_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = {"rank": "rank-2026-11", "return": "return-2026-11", "interval": "interval-2026-11"}
    retrained(FIRST, built)
    tried("interval-2026-11")
    day = FIRST + timedelta(days=1)
    swaps: list[tuple[str, date, Path]] = []

    def promoted(run_key: str, on: date, directory: Path) -> Promoted:
        swaps.append((run_key, on, directory))
        return Promoted(run_key == "rank-2026-11", [])

    monkeypatch.setattr(tasks, "first_day", lambda run_key: day)
    monkeypatch.setattr(tasks, "promote", promoted)

    summary = tasks.STEPS["promote"](day)

    assert swaps == [
        ("rank-2026-11", day, settings.FEATURE_PARQUET_DIR),
        ("return-2026-11", day, settings.FEATURE_PARQUET_DIR),
    ]
    assert summary == "rank-2026-11: promoted; return-2026-11: kept the serving runs"


def test_a_promote_that_fails_still_tries_the_others_and_fails_the_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrained(FIRST, {"rank": "rank-2026-11", "return": "return-2026-11"})
    day = FIRST + timedelta(days=1)
    swaps: list[str] = []

    def promoted(run_key: str, on: date, directory: Path) -> Promoted:
        swaps.append(run_key)
        if run_key == "rank-2026-11":
            raise RunError("the serving run of rank 20 日 changed since its gates were checked")
        return Promoted(True, [])

    monkeypatch.setattr(tasks, "first_day", lambda run_key: day)
    monkeypatch.setattr(tasks, "promote", promoted)

    with pytest.raises(RuntimeError, match="rank-2026-11: RunError: the serving run of rank 20"):
        tasks.STEPS["promote"](day)

    assert swaps == ["rank-2026-11", "return-2026-11"]


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
    assert sent == [(WEDNESDAY, tasks.MORNING_CHECK, Severity.WARNING)]
    assert {day for _, day, _ in ran} == {thursday}


@pytest.mark.parametrize("earlier", [False, True])
def test_no_missed_warning_on_the_first_morning_or_after_a_run(
    now: Callable[[date, time], None],
    gate: list[bool],
    ran: list[tuple[str, date, bool]],
    sent: list[Any],
    queued: list[str],
    earlier: bool,
) -> None:
    if earlier:
        record(WEDNESDAY, tasks.MORNING_CHECK, Severity.WARNING)
    morning_of(WEDNESDAY + timedelta(days=1), now)

    tasks.morning.apply().get()

    assert sent == []
    if earlier:
        assert marker(WEDNESDAY, tasks.MORNING_CHECK).detail == {}
    else:
        assert not DataQualityLog.objects.filter(date=WEDNESDAY, check_name=tasks.MORNING_CHECK)
