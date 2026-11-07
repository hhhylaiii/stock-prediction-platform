"""Model swaps (SPEC §7 每月重訓): promote a build past its gates, restore a degraded run."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from django.db import transaction

from apps.api.serving import day_of
from apps.etl.alerts import dispatch
from apps.etl.models import DataQualityLog
from apps.market.calendar import TradingDays
from apps.research.baseline_runs import SCOPES
from apps.research.gates import SlotGates, build_runs, check_gates
from apps.research.holdout import production_pass
from apps.research.models import ModelRun, ModelStatus
from apps.research.monitor import DEGRADED_ON
from apps.research.predict import Slot, predict, serving_run
from apps.research.runs import PRODUCTION, SCREENING_MODEL, RunError

PROMOTED_ON = "promoted_on"
PROMOTION = "promotion"
RESTORED_ON = "restored_on"
ALERT_PERIODS = "alert_periods"
PROMOTION_CHECK = "model_promotion"


@dataclass(frozen=True)
class Promoted:
    promoted: bool
    checked: list[SlotGates]
    retired: list[str] = field(default_factory=list)
    written: dict[str, int] = field(default_factory=dict)
    failed: list[dict[str, Any]] = field(default_factory=list)


def _scope(run_key: str) -> str:
    for scope in SCOPES:
        if run_key.startswith(f"{scope}-"):
            return scope
    raise RunError(f"{run_key} names no scope; production keys start with one of {list(SCOPES)}")


def _trading(day: date) -> None:
    if not TradingDays.from_db().is_trading_day(day):
        raise RunError(f"{day} is not a trading day")


def build_day(run_key: str) -> date:
    """The as_of the build of run_key was fitted through."""
    notes = [run.params[PRODUCTION] for run in build_runs(run_key).values()]
    return max(date.fromisoformat(note["as_of"]) for note in notes)


def _check_build(run_key: str, day: date, frozen_by: str) -> dict[Slot, ModelRun]:
    """N of run_key, built under the scope's current claim and as of day or earlier."""
    runs = build_runs(run_key)
    claims = sorted({run.params[PRODUCTION]["frozen_by"] for run in runs.values()} - {frozen_by})
    if claims:
        raise RunError(
            f"{run_key} was built under {', '.join(claims)}, not under the claim {frozen_by}"
        )
    as_of = build_day(run_key)
    if day < as_of:
        raise RunError(f"{run_key} is built as of {as_of}; promote it on that day or later")
    return runs


def _failed(checked: list[SlotGates]) -> list[dict[str, Any]]:
    return [
        {
            "slot": str(result.slot),
            "run_id": result.run_id,
            "gates": [
                {"name": gate.name, "reason": gate.reason}
                for gate in result.gates
                if gate.passed is False
            ],
        }
        for result in checked
        if not result.passed
    ]


def _alert(day: date, run_key: str, failed: list[dict[str, Any]], reason: str) -> DataQualityLog:
    """The warning that a build failed its gates, kept out or forced in with a reason."""
    detail = {
        "run_key": run_key,
        "forced": bool(reason),
        "reason": reason or None,
        "failed": failed,
    }
    return DataQualityLog.objects.create(
        date=day,
        check_name=PROMOTION_CHECK,
        severity=DataQualityLog.Severity.WARNING,
        detail=detail,
    )


def _swap(
    runs: dict[Slot, ModelRun], day: date, failed: list[dict[str, Any]], reason: str
) -> list[str]:
    """Retire each slot's serving run and make N active from day, noting how it got there."""
    retired = []
    for slot, run in runs.items():
        old = serving_run(slot)
        if old is not None:
            ModelRun.objects.filter(pk=old.pk).update(status=ModelStatus.RETIRED)
            retired.append(old.run_id)
        current = ModelRun.objects.get(pk=run.pk)
        note = {
            "forced": bool(failed),
            "reason": reason or None,
            "failed": [item for item in failed if item["run_id"] == run.run_id],
        }
        params = {**current.params, PROMOTED_ON: day.isoformat(), PROMOTION: note}
        ModelRun.objects.filter(pk=run.pk).update(status=ModelStatus.ACTIVE, params=params)
    return sorted(retired)


def promote(
    run_key: str, day: date, directory: Path, force: bool = False, reason: str = ""
) -> Promoted:
    """Swap build run_key in from day when every gate passes, or by force with a reason."""
    reason = reason.strip()
    if force and not reason:
        raise RunError("--force needs a --reason saying why the failed gates are overridden")
    if reason and not force:
        raise RunError("--reason goes with --force only")
    granted = production_pass(_scope(run_key))
    _trading(day)
    alert = None
    with transaction.atomic():
        runs = _check_build(run_key, day, granted.frozen_by)
        checked = check_gates(run_key, directory)
        failed = _failed(checked)
        if failed and not force:
            alert = _alert(day, run_key, failed, "")
            result = Promoted(False, checked, failed=failed)
        else:
            retired = _swap(runs, day, failed, reason)
            written = predict(day, directory, replace=True, slots=list(runs)).written
            alert = _alert(day, run_key, failed, reason) if failed else None
            result = Promoted(True, checked, retired, written, failed)
    if alert is not None:
        dispatch([alert])
    return result


def _active_in(slot: Slot, run_id: str) -> list[str]:
    runs = ModelRun.objects.filter(
        objective=slot.objective, params__horizon=slot.horizon, status=ModelStatus.ACTIVE
    ).exclude(model_type=SCREENING_MODEL)
    return sorted(runs.exclude(run_id=run_id).values_list("run_id", flat=True))


def restore(run_id: str, day: date, reason: str) -> ModelRun:
    """Put a degraded run back to active from day; its degraded days become a closed period."""
    reason = reason.strip()
    if not reason:
        raise RunError("restoring a run needs a reason")
    _trading(day)
    with transaction.atomic():
        run = ModelRun.objects.select_for_update().filter(run_id=run_id).first()
        if run is None:
            raise RunError(f"no run {run_id}")
        if run.status != ModelStatus.DEGRADED:
            raise RunError(f"{run_id} is {run.status}; only a degraded run is restored")
        since = day_of(run.params.get(DEGRADED_ON))
        if since is None:
            raise RunError(f"{run_id} has no readable degraded_on; fix it before restoring")
        if day < since:
            raise RunError(f"{run_id} was degraded on {since}, after {day}")
        slot = Slot(run.objective, int(run.params["horizon"]))
        active = _active_in(slot, run_id)
        if active:
            raise RunError(f"{', '.join(active)} is active for {slot}; retire it first")
        periods = run.params.get(ALERT_PERIODS, [])
        if not isinstance(periods, list):
            raise RunError(f"{run_id} alert_periods is not a list; fix it before restoring")
        entry = {"start": since.isoformat(), "end": day.isoformat(), "reason": reason}
        kept = {name: value for name, value in run.params.items() if name != DEGRADED_ON}
        run.status = ModelStatus.ACTIVE
        run.params = {**kept, ALERT_PERIODS: [*periods, entry], RESTORED_ON: day.isoformat()}
        run.save(update_fields=["status", "params"])
    return run
