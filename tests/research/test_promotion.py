from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.models import DataQualityLog
from apps.research import gates, promotion
from apps.research.holdout import HoldoutError
from apps.research.models import HoldoutRun, ModelRun, ModelStatus, Prediction
from apps.research.predict import PredictionGateError, Slot
from apps.research.promotion import PROMOTION_CHECK, promote
from apps.research.runs import RunError
from tests.research.test_gates import LAST, built, other_shadow, serve_copy
from tests.research.test_predict import checked, passed

RANK_SLOTS = [Slot("rank", 5), Slot("rank", 20)]


def fresh(run: ModelRun) -> ModelRun:
    return ModelRun.objects.get(pk=run.pk)


def alerts() -> list[DataQualityLog]:
    return list(DataQualityLog.objects.filter(check_name=PROMOTION_CHECK))


def sent(monkeypatch: pytest.MonkeyPatch) -> list[DataQualityLog]:
    captured: list[DataQualityLog] = []
    monkeypatch.setattr(promotion, "dispatch", lambda rows: captured.extend(rows))
    return captured


def serving_copies(runs: list[ModelRun], monkeypatch: pytest.MonkeyPatch) -> list[ModelRun]:
    monkeypatch.setattr(gates, "GAIN_TOP", 2)
    monkeypatch.setattr(gates, "GAIN_OVERLAP", 2)
    return [serve_copy(fresh(run)) for run in runs]


def untouched(runs: list[ModelRun], status: str = ModelStatus.CANDIDATE) -> None:
    for run in runs:
        current = fresh(run)
        assert current.status == status
        assert "promoted_on" not in current.params
        assert "promotion" not in current.params


@pytest.mark.django_db
def test_a_first_launch_goes_live_with_its_rows(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    dispatched = sent(monkeypatch)

    result = promote("rank-prod", LAST, researched)

    assert result.promoted is True
    assert result.retired == []
    for run in map(fresh, runs):
        assert run.status == ModelStatus.ACTIVE
        assert run.params["promoted_on"] == LAST.isoformat()
        assert run.params["promotion"] == {"forced": False, "reason": None, "failed": []}
        assert run.params["gates"]["passed"] is True
        rows = Prediction.objects.filter(run=run, date=LAST)
        assert rows.count() == result.written[run.run_id] > 0
    assert set(result.written) == {run.run_id for run in runs}
    assert not alerts()
    assert dispatched == []
    shadows = ModelRun.objects.filter(run_key="rank-prod-shadow")
    assert set(shadows.values_list("status", flat=True)) == {ModelStatus.CANDIDATE}


@pytest.mark.django_db
def test_a_promotion_retires_the_serving_runs_and_replaces_their_rows(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    old = serving_copies(runs, monkeypatch)
    ModelRun.objects.filter(pk=old[0].pk).update(status=ModelStatus.DEGRADED)

    result = promote("rank-prod", LAST, researched)

    assert result.promoted is True
    assert sorted(result.retired) == sorted(run.run_id for run in old)
    assert {fresh(run).status for run in old} == {ModelStatus.RETIRED}
    assert not Prediction.objects.filter(run__in=old).exists()
    on_day = set(Prediction.objects.filter(date=LAST).values_list("run_id", flat=True))
    assert on_day == {run.run_id for run in runs}
    for slot in RANK_SLOTS:
        live = ModelRun.objects.filter(
            status=ModelStatus.ACTIVE, objective=slot.objective, params__horizon=slot.horizon
        )
        assert live.count() == 1
    assert {fresh(run).params["gates"]["serving"] for run in runs} == {run.run_id for run in old}


@pytest.mark.django_db
def test_failed_gates_keep_the_serving_runs_and_alert(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    old = serving_copies(runs, monkeypatch)
    monkeypatch.setattr(gates, "MIN_DAYS", 500)
    dispatched = sent(monkeypatch)

    result = promote("rank-prod", LAST, researched)

    assert result.promoted is False
    assert result.written == {}
    untouched(runs)
    assert {fresh(run).status for run in old} == {ModelStatus.ACTIVE}
    assert {fresh(run).params["gates"]["passed"] for run in runs} == {False}
    assert not Prediction.objects.filter(run__in=runs).exists()
    [alert] = alerts()
    assert dispatched == [alert]
    assert alert.date == LAST
    assert alert.severity == DataQualityLog.Severity.WARNING
    assert alert.detail["run_key"] == "rank-prod"
    assert alert.detail["forced"] is False
    failed = {entry["slot"]: entry for entry in alert.detail["failed"]}
    assert set(failed) == {str(slot) for slot in RANK_SLOTS}
    for entry in failed.values():
        assert [gate["name"] for gate in entry["gates"]] == ["shadow_oos"]
        assert "少於 500" in entry["gates"][0]["reason"]


@pytest.mark.django_db
def test_force_swaps_in_a_build_that_failed_and_records_why(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    monkeypatch.setattr(gates, "MIN_DAYS", 500)
    dispatched = sent(monkeypatch)
    why = "the shadow window is short on this panel"

    result = promote("rank-prod", LAST, researched, force=True, reason=f"  {why} ")

    assert result.promoted is True
    for run in map(fresh, runs):
        assert run.status == ModelStatus.ACTIVE
        note = run.params["promotion"]
        assert note["forced"] is True
        assert note["reason"] == why
        [failed] = note["failed"]
        assert failed["run_id"] == run.run_id
        assert [gate["name"] for gate in failed["gates"]] == ["shadow_oos"]
    [alert] = alerts()
    assert dispatched == [alert]
    assert alert.detail["forced"] is True
    assert alert.detail["reason"] == why
    assert len(alert.detail["failed"]) == 2


@pytest.mark.django_db
def test_force_without_failures_is_not_recorded_as_forced(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    sent(monkeypatch)

    promote("rank-prod", LAST, researched, force=True, reason="routine")

    assert {fresh(run).params["promotion"]["forced"] for run in runs} == {False}
    assert not alerts()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("force", "reason", "match"),
    [
        (True, "", "--force needs a --reason"),
        (True, "   ", "--force needs a --reason"),
        (False, "why", "--reason goes with --force"),
    ],
)
def test_force_and_reason_go_together(
    researched: Path, force: bool, reason: str, match: str
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)

    with pytest.raises(RunError, match=match):
        promote("rank-prod", LAST, researched, force=force, reason=reason)

    untouched(runs)
    assert not ModelRun.objects.filter(run_key="rank-prod", params__has_key="gates").exists()


def saturday_before(day: date) -> date:
    return day - timedelta(days=(day.weekday() - 5) % 7)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("moved", RunError, r"left candidate"),
        ("other shadow", RunError, r"is the N′ of rank-prod as of 2016-11-30"),
        ("no claim", HoldoutError, r"rank design is not frozen"),
        ("other claim", RunError, r"built under rank-other, not under the claim rank-final"),
        ("early", RunError, r"is built as of"),
        ("weekend", RunError, r"is not a trading day"),
    ],
)
def test_force_never_skips_a_refused_build(
    researched: Path, case: str, error: type[Exception], match: str
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    day = LAST
    if case == "moved":
        ModelRun.objects.filter(pk=runs[0].pk).update(status=ModelStatus.ACTIVE)
    if case == "other shadow":
        other_shadow("rank-prod", as_of="2016-11-30")
    if case == "no claim":
        HoldoutRun.objects.filter(scope="rank").delete()
    if case == "other claim":
        for run in runs:
            note = {**run.params["production"], "frozen_by": "rank-other"}
            ModelRun.objects.filter(pk=run.pk).update(params={**run.params, "production": note})
    if case == "early":
        day = LAST - timedelta(days=1 if LAST.weekday() > 0 else 3)
    if case == "weekend":
        day = saturday_before(LAST)

    with pytest.raises(error, match=match):
        promote("rank-prod", day, researched, force=True, reason="overriding")

    untouched(runs[1:])
    assert not alerts()
    assert not Prediction.objects.exists()


@pytest.mark.django_db
def test_a_failed_write_rolls_the_whole_promotion_back(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    checked(LAST)
    old = serving_copies(runs, monkeypatch)
    kept = Prediction.objects.filter(run__in=old).count()

    with pytest.raises(PredictionGateError):
        promote("rank-prod", LAST, researched)

    untouched(runs)
    assert {fresh(run).status for run in old} == {ModelStatus.ACTIVE}
    assert Prediction.objects.filter(run__in=old).count() == kept > 0
    assert not ModelRun.objects.filter(run_key="rank-prod", params__has_key="gates").exists()


@pytest.mark.django_db
def test_the_command_promotes_on_the_build_day_by_default(
    researched: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    settings.FEATURE_PARQUET_DIR = researched
    sent(monkeypatch)
    out = StringIO()

    call_command("promote", run_key="rank-prod", stdout=out)

    text = out.getvalue()
    assert f"rank-prod 於 {LAST} 換模" in text
    for run in runs:
        assert run.run_id in text
    assert {fresh(run).status for run in runs} == {ModelStatus.ACTIVE}


@pytest.mark.django_db
def test_the_command_reports_failed_gates_and_its_errors(
    researched: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    built("rank", "rank-wf1", researched)
    passed(LAST)
    settings.FEATURE_PARQUET_DIR = researched
    monkeypatch.setattr(gates, "MIN_DAYS", 500)
    sent(monkeypatch)
    out = StringIO()

    call_command("promote", run_key="rank-prod", date=LAST.isoformat(), stdout=out)

    assert "未換模，舊模型續用" in out.getvalue()
    assert "shadow_oos" in out.getvalue()
    with pytest.raises(CommandError, match="--force needs a --reason"):
        call_command("promote", run_key="rank-prod", force=True)
    with pytest.raises(CommandError, match="holds no production model N"):
        call_command("promote", run_key="rank-wf1")
