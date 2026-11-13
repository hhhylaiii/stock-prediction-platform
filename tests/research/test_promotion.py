from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.models import DataQualityLog
from apps.research import gates, promotion
from apps.research.gates import Gate, SlotGates
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
            reason = f"1 天有 error，最近 {LAST} row_count"
            errors = {
                "errors": [{"date": LAST.isoformat(), "check": "row_count"}],
                "errors_total": 1,
            }
            return Gate("data_quality", False, 1.0, "無 error", 1, reason, errors)
        return checked_quality(run)

    monkeypatch.setattr(gates, "data_quality", quality)


def failing_objective(monkeypatch: pytest.MonkeyPatch, objective: str) -> None:
    checked_quality = gates.data_quality

    def quality(run: ModelRun) -> Gate:
        if run.objective == objective:
            return Gate("data_quality", False, 1.0, "無 error", 1, "1 天有 error", {})
        return checked_quality(run)

    monkeypatch.setattr(gates, "data_quality", quality)


def by_horizon(runs: list[ModelRun]) -> list[ModelRun]:
    return sorted(runs, key=lambda run: (int(run.params["horizon"]), run.objective))


def entry(run: ModelRun, **extra: Any) -> dict[str, Any]:
    """A run as the promotion record lists it among the swapped or the held."""
    horizon = int(run.params["horizon"])
    return {
        "slot": f"{run.objective} {horizon} 日",
        "objective": run.objective,
        "horizon": horizon,
        "run_id": run.run_id,
        **extra,
    }


QUALITY_GATE = {
    "name": "data_quality",
    "reason": f"1 天有 error，最近 {LAST} row_count",
    "value": 1.0,
    "threshold": "無 error",
    "sample": 1,
    "sample_unit": "days",
}


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
    [record] = alerts()
    assert (record.date, record.severity) == (LAST, DataQualityLog.Severity.INFO)
    assert record.detail == {
        "scope": "rank",
        "run_key": "rank-prod",
        "forced": False,
        "reason": None,
        "failed": [],
        "promoted": True,
        "promoted_slots": [entry(run, forced=False) for run in by_horizon(runs)],
        "held_slots": [],
    }
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
def test_a_promotion_rechecks_the_outputs_it_rewrote_and_a_kept_build_does_not(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    serving_copies(runs, monkeypatch)
    rechecked: list[date] = []
    monkeypatch.setattr(promotion, "recheck_outputs", rechecked.append)
    default = gates.MIN_DAYS
    monkeypatch.setattr(gates, "MIN_DAYS", 500)

    promote("rank-prod", LAST, researched)

    assert rechecked == []
    monkeypatch.setattr(gates, "MIN_DAYS", default)

    promote("rank-prod", LAST, researched)

    assert rechecked == [LAST]


@pytest.mark.django_db
def test_a_recheck_that_fails_after_the_swap_is_logged_and_the_swap_stands(
    researched: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    serving_copies(runs, monkeypatch)

    def broken(day: date) -> list[DataQualityLog]:
        raise RuntimeError("no outputs frame")

    monkeypatch.setattr(promotion, "recheck_outputs", broken)

    result = promote("rank-prod", LAST, researched)

    assert result.promoted is True
    assert {fresh(run).status for run in runs} == {ModelStatus.ACTIVE}
    assert "rank-prod" in caplog.text
    assert "no outputs frame" in caplog.text


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
    assert (alert.detail["scope"], alert.detail["run_key"]) == ("rank", "rank-prod")
    assert alert.detail["forced"] is False
    assert alert.detail["promoted"] is False
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
    assert alert.detail["promoted"] is True
    assert alert.detail["scope"] == "rank"
    assert alert.severity == DataQualityLog.Severity.WARNING
    assert len(alert.detail["failed"]) == 2


@pytest.mark.django_db
def test_force_without_failures_is_not_recorded_as_forced(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    dispatched = sent(monkeypatch)

    promote("rank-prod", LAST, researched, force=True, reason="routine")

    for run in map(fresh, runs):
        assert run.params["promotion"] == {"forced": False, "reason": None, "failed": []}
    [record] = alerts()
    assert record.severity == DataQualityLog.Severity.INFO
    assert record.detail == {
        "scope": "rank",
        "run_key": "rank-prod",
        "forced": False,
        "reason": None,
        "failed": [],
        "promoted": True,
        "promoted_slots": [entry(run, forced=False) for run in by_horizon(runs)],
        "held_slots": [],
    }
    assert dispatched == []


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
        ModelRun.objects.filter(pk__in=[run.pk for run in runs]).update(status=ModelStatus.ACTIVE)
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

    untouched(runs, ModelStatus.ACTIVE if case == "moved" else ModelStatus.CANDIDATE)
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
    assert [row.detail["promoted"] for row in sorted(alerts(), key=lambda row: row.pk)] == [
        False,
        True,
    ]


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
def test_a_failing_horizon_stays_out_while_the_passing_one_goes_live(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    failing_h5(monkeypatch)
    dispatched = sent(monkeypatch)
    five, twenty = by_horizon(runs)

    result = promote("rank-prod", LAST, researched)

    assert result.promoted is True
    assert result.held == ["rank 5 日"]
    assert set(result.written) == {twenty.run_id}
    assert fresh(twenty).status == ModelStatus.ACTIVE
    assert fresh(twenty).params["promoted_on"] == LAST.isoformat()
    assert fresh(twenty).params["promotion"] == {"forced": False, "reason": None, "failed": []}
    untouched([five])
    assert fresh(five).params["gates"]["passed"] is False
    assert set(Prediction.objects.values_list("run_id", "horizon")) == {(twenty.run_id, 20)}
    [alert] = alerts()
    assert dispatched == [alert]
    assert alert.severity == DataQualityLog.Severity.WARNING
    assert alert.detail == {
        "scope": "rank",
        "run_key": "rank-prod",
        "forced": False,
        "reason": None,
        "failed": [{"slot": "rank 5 日", "run_id": five.run_id, "gates": [QUALITY_GATE]}],
        "promoted": True,
        "promoted_slots": [entry(twenty, forced=False)],
        "held_slots": [entry(five, gates=[QUALITY_GATE])],
    }


@pytest.mark.django_db
def test_only_the_horizon_swapped_in_retires_its_serving_run(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    old_five, old_twenty = by_horizon(serving_copies(runs, monkeypatch))
    failing_h5(monkeypatch)
    sent(monkeypatch)
    five, twenty = by_horizon(runs)
    kept = set(Prediction.objects.filter(run=old_five, date=LAST).values_list("pk", flat=True))

    result = promote("rank-prod", LAST, researched)

    assert result.retired == [old_twenty.run_id]
    assert set(result.written) == {twenty.run_id}
    assert fresh(old_twenty).status == ModelStatus.RETIRED
    assert fresh(old_five).status == ModelStatus.ACTIVE
    rows = Prediction.objects.filter(run=old_five, date=LAST)
    assert set(rows.values_list("pk", flat=True)) == kept != set()
    assert not Prediction.objects.filter(run=old_twenty, date=LAST).exists()
    assert not Prediction.objects.filter(run=five).exists()
    assert Prediction.objects.filter(run=twenty, date=LAST).count() == result.written[twenty.run_id]


@pytest.mark.django_db
def test_a_held_horizon_comes_in_later_only_by_force(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    failing_h5(monkeypatch)
    sent(monkeypatch)
    five, twenty = by_horizon(runs)
    promote("rank-prod", LAST, researched)
    served = fresh(twenty).params

    again = promote("rank-prod", LAST, researched)

    assert again.promoted is False
    assert again.held == ["rank 5 日"]
    assert [result.slot for result in again.checked] == [Slot("rank", 5)]
    untouched([five])
    assert fresh(twenty).params == served
    assert alerts()[-1].detail["promoted"] is False
    assert alerts()[-1].detail["promoted_slots"] == []
    assert alerts()[-1].detail["held_slots"] == [entry(five, gates=[QUALITY_GATE])]

    forced = promote("rank-prod", LAST, researched, force=True, reason="checked by hand")

    assert forced.promoted is True
    assert forced.held == []
    assert forced.retired == []
    assert set(forced.written) == {five.run_id}
    assert fresh(five).status == ModelStatus.ACTIVE
    assert fresh(five).params["promotion"]["forced"] is True
    assert fresh(five).params["promotion"]["reason"] == "checked by hand"
    assert (fresh(twenty).status, fresh(twenty).params) == (ModelStatus.ACTIVE, served)
    assert Prediction.objects.filter(run=twenty, date=LAST).exists()
    last = alerts()[-1].detail
    assert (last["promoted"], last["forced"], last["reason"]) == (True, True, "checked by hand")
    assert last["promoted_slots"] == [entry(five, forced=True)]
    assert last["held_slots"] == []
    assert len(alerts()) == 3


@pytest.mark.django_db
def test_force_swaps_in_every_horizon_and_marks_only_the_failed_one_forced(
    researched: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    failing_h5(monkeypatch)
    settings.FEATURE_PARQUET_DIR = researched
    sent(monkeypatch)
    five, twenty = by_horizon(runs)
    out = StringIO()

    call_command("promote", run_key="rank-prod", force=True, reason="checked by hand", stdout=out)

    assert "強制換模（checked by hand）" in out.getvalue()
    assert f"{five.run_id}：data_quality（1 天有 error，最近 {LAST} row_count）" in out.getvalue()
    assert fresh(five).params["promotion"]["forced"] is True
    assert fresh(five).params["promotion"]["failed"][0]["gates"] == [QUALITY_GATE]
    assert fresh(twenty).params["promotion"] == {
        "forced": False,
        "reason": "checked by hand",
        "failed": [],
    }
    assert {fresh(run).status for run in runs} == {ModelStatus.ACTIVE}
    [alert] = alerts()
    assert alert.detail["promoted_slots"] == [entry(five, forced=True), entry(twenty, forced=False)]
    assert alert.detail["held_slots"] == []


@pytest.mark.django_db
def test_the_command_names_the_horizon_it_left_out(
    researched: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    failing_h5(monkeypatch)
    settings.FEATURE_PARQUET_DIR = researched
    sent(monkeypatch)
    five, twenty = by_horizon(runs)
    out = StringIO()

    call_command("promote", run_key="rank-prod", stdout=out)

    lines = out.getvalue().splitlines()
    assert lines[0] == f"rank-prod 於 {LAST} 換模"
    assert lines[1].startswith(f"  {twenty.run_id}：active，寫入 ")
    assert lines[2] == "  沒過門檻、未換上（舊模型續用）："
    assert lines[3] == (
        f"  rank 5 日 {five.run_id}：data_quality（1 天有 error，最近 {LAST} row_count）"
    )
    assert len(lines) == 4
    assert "強制換模" not in out.getvalue()


@pytest.mark.django_db
def test_the_gate_report_says_which_horizon_could_go_in_alone(
    researched: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    built("rank", "rank-wf1", researched)
    passed(LAST)
    failing_h5(monkeypatch)
    settings.FEATURE_PARQUET_DIR = researched
    out = StringIO()

    call_command("check_gates", run_key="rank-prod", stdout=out)

    assert out.getvalue().strip().endswith("未通過：rank 5 日；可單獨換上：rank 20 日")
    assert not ModelRun.objects.exclude(status=ModelStatus.CANDIDATE).exists()


@pytest.mark.django_db
def test_the_three_interval_models_go_live_together_or_not_at_all(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("interval", "interval-wf1", researched)
    passed(LAST)
    clear = Gate("shadow_oos", True, 0.0, "", gates.MIN_DAYS, "")
    monkeypatch.setattr(gates, "shadow_oos", lambda run, shadow, directory: clear)
    failing_objective(monkeypatch, "volatility")
    sent(monkeypatch)
    volatility = next(run for run in runs if run.objective == "volatility")
    gate = {**QUALITY_GATE, "reason": "1 天有 error"}

    result = promote("interval-prod", LAST, researched)

    assert result.promoted is False
    assert result.held == ["classification 20 日", "quantile 20 日", "volatility 20 日"]
    untouched(runs)
    assert not Prediction.objects.exists()
    [alert] = alerts()
    assert alert.detail["promoted_slots"] == []
    assert alert.detail["held_slots"] == [
        entry(run, gates=[gate] if run == volatility else []) for run in by_horizon(runs)
    ]
    assert [item["run_id"] for item in alert.detail["failed"]] == [volatility.run_id]

    promote("interval-prod", LAST, researched, force=True, reason="checked by hand")

    assert {fresh(run).status for run in runs} == {ModelStatus.ACTIVE}
    assert alerts()[-1].detail["promoted_slots"] == [
        entry(run, forced=run == volatility) for run in by_horizon(runs)
    ]


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


@pytest.mark.django_db
def test_rows_a_held_horizon_gains_while_the_gates_run_do_not_stop_the_other(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    passed(LAST)
    failing_h5(monkeypatch)
    sent(monkeypatch)
    five, twenty = by_horizon(runs)
    judge = promotion.check_gates

    def judged_while_predicting(run_key: str, directory: Path) -> list[Any]:
        found = judge(run_key, directory)
        [late] = retired_copies([five])
        Prediction.objects.filter(run=late).update(date=LAST + timedelta(days=1))
        return found

    monkeypatch.setattr(promotion, "check_gates", judged_while_predicting)

    result = promote("rank-prod", LAST, researched)

    assert set(result.written) == {twenty.run_id}
    assert fresh(twenty).status == ModelStatus.ACTIVE
    untouched([five])


def test_failed_gates_carry_the_unit_of_their_sample() -> None:
    checked = [
        SlotGates(
            Slot("rank", 20),
            "rank-prod.lightgbm.h20.f000",
            "rank-old.lightgbm.h20.f000",
            [
                Gate("agreement", False, 0.4, ">= 0.7", 38, "相關太低"),
                Gate("gain_overlap", False, 3.0, "前 10 名重疊 >= 6", 12, "重疊太少"),
                Gate("shadow_oos", True, 0.02, "rank IC > 0", 120, ""),
            ],
        )
    ]

    [failed] = promotion._failed(checked)

    assert [(gate["name"], gate["sample"], gate["sample_unit"]) for gate in failed["gates"]] == [
        ("agreement", 38, "stocks"),
        ("gain_overlap", 12, "features"),
    ]
