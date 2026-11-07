from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.models import DataQualityLog
from apps.research import gates, promotion
from apps.research.gates import Gate
from apps.research.holdout import HoldoutError
from apps.research.models import HoldoutRun, ModelRun, ModelStatus, Prediction
from apps.research.predict import PredictionGateError, Slot
from apps.research.promotion import PROMOTION_CHECK, promote
from apps.research.runs import RunError
from tests.research.run_panels import weekdays
from tests.research.test_gates import LAST, built, logged, other_shadow, serve_copy
from tests.research.test_predict import checked, passed

RANK_SLOTS = [Slot("rank", 5), Slot("rank", 20)]
PREV = weekdays()[-2]


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


def rebased(run_key: str, as_of: date) -> None:
    """Move a build's as_of, for N and N′ alike, as if it had been fitted a day earlier."""
    for run in ModelRun.objects.filter(run_key__in=[run_key, f"{run_key}-shadow"]):
        note = {**run.params["production"], "as_of": as_of.isoformat()}
        ModelRun.objects.filter(pk=run.pk).update(params={**run.params, "production": note})


def retired_copies(runs: list[ModelRun]) -> list[ModelRun]:
    """Retired runs that served the build's slots on LAST."""
    copies = [serve_copy(fresh(run)) for run in runs]
    ModelRun.objects.filter(pk__in=[run.pk for run in copies]).update(status=ModelStatus.RETIRED)
    return copies


def failing_h5(monkeypatch: pytest.MonkeyPatch) -> None:
    checked_quality = gates.data_quality

    def quality(run: ModelRun) -> Gate:
        if run.params["horizon"] == 5:
            return Gate("data_quality", False, 1.0, "no error", 1, f"{LAST} row_count")
        return checked_quality(run)

    monkeypatch.setattr(gates, "data_quality", quality)


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
    others = built("interval", "interval-wf1", researched)
    ModelRun.objects.filter(pk__in=[run.pk for run in others]).update(status=ModelStatus.ACTIVE)
    elsewhere = others[0]
    Prediction.objects.create(
        run=elsewhere, stock_id="1101", date=LAST, horizon=20, in_universe=True, vol_pred=0.3
    )
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    dispatched = sent(monkeypatch)

    result = promote("rank-prod", LAST, researched)

    assert result.promoted is True
    assert result.retired == []
    assert set(
        Prediction.objects.filter(date=LAST, horizon=20)
        .exclude(run__objective="rank")
        .values_list("run_id", "stock_id")
    ) == {(elsewhere.run_id, "1101")}
    assert {fresh(run).status for run in others} == {ModelStatus.ACTIVE}
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
        [gate] = entry["gates"]
        assert gate["name"] == "shadow_oos"
        assert "少於 500" in gate["reason"]
        assert gate["sample_unit"] == "days"
        assert 0 < gate["sample"] < 500
        assert gate["threshold"] == "rank IC > 0"
        assert isinstance(gate["value"], float)


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


@pytest.mark.django_db
def test_a_late_promotion_serves_and_alerts_from_its_own_day(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    rebased("rank-prod", PREV)
    logged(runs[0])
    passed(LAST)
    sent(monkeypatch)
    monkeypatch.setattr(gates, "MIN_DAYS", 500)

    kept = promote("rank-prod", LAST, researched)

    assert kept.promoted is False
    assert [alert.date for alert in alerts()] == [LAST]

    monkeypatch.setattr(gates, "MIN_DAYS", 1)
    result = promote("rank-prod", LAST, researched)

    assert result.promoted is True
    for run in map(fresh, runs):
        assert run.params["promoted_on"] == LAST.isoformat()
        assert set(Prediction.objects.filter(run=run).values_list("date", flat=True)) == {LAST}


@pytest.mark.django_db
def test_later_rows_set_the_default_day_and_refuse_an_earlier_one(
    researched: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    rebased("rank-prod", PREV)
    logged(runs[0])
    passed(LAST)
    copies = retired_copies(runs)
    settings.FEATURE_PARQUET_DIR = researched
    sent(monkeypatch)

    with pytest.raises(RunError, match=f"already hold rows through {LAST}; promote on {LAST}"):
        promote("rank-prod", PREV, researched)

    untouched(runs)
    assert Prediction.objects.filter(run__in=copies, date=LAST).exists()
    out = StringIO()

    call_command("promote", run_key="rank-prod", stdout=out)

    assert f"rank-prod 於 {LAST} 換模" in out.getvalue()
    assert {fresh(run).params["promoted_on"] for run in runs} == {LAST.isoformat()}
    assert not Prediction.objects.filter(run__in=copies).exists()


@pytest.mark.django_db
def test_one_failing_slot_keeps_the_whole_build_out_unless_forced(
    researched: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    failing_h5(monkeypatch)
    settings.FEATURE_PARQUET_DIR = researched
    sent(monkeypatch)
    five = next(run for run in runs if run.params["horizon"] == 5)
    twenty = next(run for run in runs if run.params["horizon"] == 20)

    kept = promote("rank-prod", LAST, researched)

    assert kept.promoted is False
    untouched(runs)
    assert not Prediction.objects.exists()
    [alert] = alerts()
    assert [item["run_id"] for item in alert.detail["failed"]] == [five.run_id]
    out = StringIO()

    call_command("promote", run_key="rank-prod", force=True, reason="checked by hand", stdout=out)

    assert "強制換模（checked by hand）" in out.getvalue()
    assert f"{five.run_id}：data_quality（{LAST} row_count）" in out.getvalue()
    assert fresh(five).params["promotion"]["forced"] is True
    assert fresh(five).params["promotion"]["failed"][0]["gates"] == [
        {
            "name": "data_quality",
            "reason": f"{LAST} row_count",
            "value": 1.0,
            "threshold": "no error",
            "sample": 1,
            "sample_unit": "days",
        }
    ]
    assert fresh(twenty).params["promotion"] == {
        "forced": False,
        "reason": "checked by hand",
        "failed": [],
    }
    assert {fresh(run).status for run in runs} == {ModelStatus.ACTIVE}


@pytest.mark.django_db
def test_promote_judges_fresh_gates_not_the_last_recorded_ones(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    sent(monkeypatch)
    gates.check_gates("rank-prod", researched)
    assert {fresh(run).params["gates"]["passed"] for run in runs} == {True}
    monkeypatch.setattr(gates, "MIN_DAYS", 500)

    result = promote("rank-prod", LAST, researched)

    assert result.promoted is False
    untouched(runs)
    assert {fresh(run).params["gates"]["passed"] for run in runs} == {False}


@pytest.mark.django_db
def test_a_forced_promotion_that_fails_to_write_sends_no_alert(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    checked(LAST)
    monkeypatch.setattr(gates, "MIN_DAYS", 500)
    dispatched = sent(monkeypatch)

    with pytest.raises(PredictionGateError):
        promote("rank-prod", LAST, researched, force=True, reason="overriding")

    untouched(runs)
    assert alerts() == []
    assert dispatched == []


@pytest.mark.django_db
def test_a_serving_run_that_changed_after_the_gates_stops_the_swap(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    sent(monkeypatch)
    judge = promotion.check_gates

    def judged_then_moved(run_key: str, directory: Path) -> list[Any]:
        found = judge(run_key, directory)
        copy = serve_copy(fresh(runs[0]))
        ModelRun.objects.filter(pk=copy.pk).update(status=ModelStatus.ACTIVE)
        return found

    monkeypatch.setattr(promotion, "check_gates", judged_then_moved)
    with pytest.raises(RunError, match="changed since its gates were checked"):
        promote("rank-prod", LAST, researched)

    untouched(runs)


@pytest.mark.django_db
def test_every_serving_run_of_a_slot_is_retired(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    old = serving_copies(runs, monkeypatch)
    stale = fresh(old[0])
    stale.run_id, stale.status = f"{stale.run_id}.stale", ModelStatus.DEGRADED
    stale.save(force_insert=True)

    result = promote("rank-prod", LAST, researched)

    assert stale.run_id in result.retired
    assert fresh(stale).status == ModelStatus.RETIRED
    assert {fresh(run).status for run in old} == {ModelStatus.RETIRED}


@pytest.mark.django_db
def test_rows_written_past_the_day_while_the_gates_run_stop_the_swap(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    rebased("rank-prod", PREV)
    logged(runs[0])
    passed(PREV)
    sent(monkeypatch)
    judge = promotion.check_gates

    def judged_while_predicting(run_key: str, directory: Path) -> list[Any]:
        found = judge(run_key, directory)
        retired_copies(runs)
        return found

    monkeypatch.setattr(promotion, "check_gates", judged_while_predicting)
    with pytest.raises(RunError, match=f"already hold rows through {LAST}"):
        promote("rank-prod", PREV, researched)

    untouched(runs)
    assert not Prediction.objects.exists()
