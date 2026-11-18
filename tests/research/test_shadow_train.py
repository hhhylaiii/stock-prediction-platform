from collections.abc import Callable
from datetime import date, time, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from django.conf import settings
from django.test import override_settings

from apps.etl.models import DataQualityLog
from apps.research import tasks
from apps.research.lock import ResearchBusyError
from tests.research.test_tasks import Severity, fast, now, retries_run_in_place, sent

pytestmark = pytest.mark.django_db
FIXTURES = (fast, now, retries_run_in_place, sent)
DAY = date(2026, 12, 3)
FROZEN = SimpleNamespace(
    serving_run_id="rank-2026-12.lightgbm.h5.f000",
    generation=1,
    first_signal=date(2026, 12, 4),
    recipe={"seconds": 12.5, "peak_mb": 900.0},
)


@pytest.fixture(autouse=True)
def morning(now: Callable[[date, time], None]) -> None:
    now(DAY + timedelta(days=1), time(6, 45))


@pytest.fixture(autouse=True)
def memory(monkeypatch: pytest.MonkeyPatch) -> list[float | None]:
    readings: list[float | None] = [80.0]
    monkeypatch.setattr(tasks, "free_percent", lambda: readings[0])
    return readings


@pytest.fixture(autouse=True)
def behind(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    names = ["C1", "C2"]
    monkeypatch.setattr(tasks, "unfollowed", lambda: list(names))
    return names


@pytest.fixture
def freezes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Each candidate the fit froze, or the error a test makes it raise."""
    state: dict[str, Any] = {"called": [], "raise": {}}

    def fake(name: str, directory: Any, replace: bool = False) -> Any:
        state["called"].append(name)
        if name in state["raise"]:
            raise state["raise"][name]
        return FROZEN

    monkeypatch.setattr(tasks, "freeze", fake)
    return state


def train() -> Any:
    return tasks.shadow_train.apply(args=(DAY.isoformat(),)).get()


def trained() -> DataQualityLog:
    return DataQualityLog.objects.get(date=DAY, check_name=tasks.SHADOW_TRAIN_CHECK)


def test_a_morning_with_every_step_done_queues_one_fit_of_the_candidates_left_behind(
    monkeypatch: pytest.MonkeyPatch, behind: list[str]
) -> None:
    queued: list[tuple[Any, ...]] = []

    def queue(args: tuple[str], soft_time_limit: int, time_limit: int) -> None:
        queued.append((args, soft_time_limit, time_limit))

    monkeypatch.setattr(tasks, "_retrain_after", lambda day: None)
    monkeypatch.setattr(tasks.shadow_train, "apply_async", queue)
    monkeypatch.setattr(tasks, "dispatch", lambda rows: None)

    with override_settings(
        RESEARCH_SHADOW_TRAIN_SOFT_SECONDS=1200, RESEARCH_SHADOW_TRAIN_HARD_SECONDS=1500
    ):
        tasks.finish([], tasks.MORNING_CHECK, DAY.isoformat(), {})
    tasks.finish([], tasks.MORNING_CHECK, DAY.isoformat(), {"us_close": "missing"})
    tasks.finish([], tasks.EVENING_CHECK, DAY.isoformat(), {})
    behind.clear()
    tasks.finish([], tasks.MORNING_CHECK, DAY.isoformat(), {})

    assert queued == [((DAY.isoformat(),), 1200, 1500)]


def test_the_fit_freezes_each_candidate_and_records_one_info_row(
    freezes: dict[str, Any], sent: list[tuple[date, str, str]]
) -> None:
    assert train() == Severity.INFO

    assert freezes["called"] == ["C1", "C2"]
    row = trained()
    assert row.severity == Severity.INFO
    shown = {
        "serving_run": FROZEN.serving_run_id,
        "generation": 1,
        "first_signal": "2026-12-04",
        "seconds": 12.5,
        "peak_mb": 900.0,
    }
    assert row.detail == {"frozen": {"C1": shown, "C2": shown}, "failed": {}}
    assert sent == []
    assert not DataQualityLog.objects.filter(check_name=tasks.MORNING_CHECK).exists()


@pytest.mark.parametrize(
    ("hour", "free", "workers", "reason"),
    [
        (time(9, 0), 80.0, 1, "outside the 06:00 to 09:00 hours a fit may start in"),
        (time(6, 45), 65.0, 1, "65.0% of memory free, under 70%"),
        (time(6, 45), 80.0, 2, "worker concurrency 2, not 1"),
    ],
)
def test_a_fit_that_may_not_start_only_warns(
    hour: time,
    free: float,
    workers: int,
    reason: str,
    now: Callable[[date, time], None],
    memory: list[float | None],
    monkeypatch: pytest.MonkeyPatch,
    freezes: dict[str, Any],
    sent: list[tuple[date, str, str]],
) -> None:
    now(DAY + timedelta(days=1), hour)
    memory[0] = free
    monkeypatch.setattr(tasks, "_concurrency", lambda task: workers)

    assert train() == "refused"

    assert freezes["called"] == []
    assert (trained().severity, trained().detail) == (Severity.WARNING, {"refused": reason})
    assert sent == [(DAY, tasks.SHADOW_TRAIN_CHECK, Severity.WARNING)]


@override_settings(RESEARCH_SHADOW_TRAIN_MIN_FREE=60, RESEARCH_RETRAIN_MIN_FREE=90)
def test_the_fit_reads_its_own_memory_floor_not_the_retrain_s(
    memory: list[float | None], freezes: dict[str, Any], sent: list[tuple[date, str, str]]
) -> None:
    memory[0] = 75.0
    assert train() == Severity.INFO
    assert freezes["called"] == ["C1", "C2"]

    memory[0] = 55.0
    DataQualityLog.objects.all().delete()
    assert train() == "refused"
    assert trained().detail == {"refused": "55.0% of memory free, under 60%"}
    assert sent == [(DAY, tasks.SHADOW_TRAIN_CHECK, Severity.WARNING)]


def test_a_failing_candidate_warns_and_the_next_one_still_freezes(
    freezes: dict[str, Any], sent: list[tuple[date, str, str]]
) -> None:
    freezes["raise"]["C1"] = ResearchBusyError("held by build_features")

    assert train() == Severity.WARNING

    assert freezes["called"] == ["C1", "C2"]
    detail = trained().detail
    assert detail["failed"] == {"C1": "ResearchBusyError: held by build_features"}
    assert list(detail["frozen"]) == ["C2"]
    assert sent == [(DAY, tasks.SHADOW_TRAIN_CHECK, Severity.WARNING)]


def test_the_soft_time_limit_stops_the_fit_with_a_warning(freezes: dict[str, Any]) -> None:
    freezes["raise"]["C1"] = SoftTimeLimitExceeded()

    assert train() == Severity.WARNING

    assert freezes["called"] == ["C1"]
    assert trained().detail == {
        "frozen": {},
        "failed": {"C1": "stopped at the 1800 s soft limit"},
    }


def test_the_fit_carries_the_soft_and_hard_time_limits_of_the_settings() -> None:
    assert tasks.shadow_train.soft_time_limit == settings.RESEARCH_SHADOW_TRAIN_SOFT_SECONDS
    assert tasks.shadow_train.time_limit == settings.RESEARCH_SHADOW_TRAIN_HARD_SECONDS


def test_a_failing_list_of_candidates_only_warns(
    monkeypatch: pytest.MonkeyPatch, freezes: dict[str, Any]
) -> None:
    def broken() -> list[str]:
        raise ResearchBusyError("no database")

    monkeypatch.setattr(tasks, "unfollowed", broken)

    assert train() == "refused"

    assert freezes["called"] == []
    assert trained().detail == {
        "refused": "listing the candidates failed: ResearchBusyError: no database"
    }
    assert trained().severity == Severity.WARNING
