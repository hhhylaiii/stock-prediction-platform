"""Model swaps (SPEC §7 每月重訓): promote a build past its gates, restore a degraded run."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from django.db import transaction
from django.db.models import Max, Q
from django.utils import timezone

from apps.api.serving import day_of
from apps.etl.alerts import dispatch
from apps.etl.models import DataQualityLog
from apps.market.calendar import TradingDays
from apps.research.baseline_runs import SCOPES
from apps.research.gates import SAMPLE_UNITS, SlotGates, build_runs, check_gates
from apps.research.holdout import production_pass
from apps.research.lock import locked
from apps.research.models import ModelRun, ModelStatus, Prediction
from apps.research.monitor import (
    DEGRADED_ON,
    DEGRADED_WINDOW,
    PROMOTED_ON,
    RESTORED_ON,
    recheck_outputs,
)
from apps.research.predict import SERVING, Slot, predict, serving_run
from apps.research.runs import PRODUCTION, SCREENING_MODEL, RunError

PROMOTION = "promotion"
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


def _as_of(runs: dict[Slot, ModelRun]) -> date:
    return max(date.fromisoformat(run.params[PRODUCTION]["as_of"]) for run in runs.values())


def _last_rows(slots: Iterable[Slot]) -> date | None:
    """The latest day any run holds prediction rows for the slots."""
    condition = Q()
    for slot in slots:
        condition |= Q(run__objective=slot.objective, horizon=slot.horizon)
    last: date | None = Prediction.objects.filter(condition).aggregate(last=Max("date"))["last"]
    return last


def first_day(run_key: str) -> date:
    """The first day build run_key can serve: its as_of, or its slots' latest rows if later."""
    runs = build_runs(run_key)
    last = _last_rows(runs)
    return max(_as_of(runs), last) if last is not None else _as_of(runs)


def _check_build(run_key: str, day: date, frozen_by: str) -> dict[Slot, ModelRun]:
    """N of run_key, built under the current claim, as of day or earlier, with no rows past day."""
    runs = build_runs(run_key)
    claims = sorted({run.params[PRODUCTION]["frozen_by"] for run in runs.values()} - {frozen_by})
    if claims:
        raise RunError(
            f"{run_key} was built under {', '.join(claims)}, not under the claim {frozen_by}"
        )
    as_of = _as_of(runs)
    if day < as_of:
        raise RunError(f"{run_key} is built as of {as_of}; promote it on that day or later")
    _refuse_later_rows(run_key, runs, day)
    return runs


def _refuse_later_rows(run_key: str, runs: dict[Slot, ModelRun], day: date) -> None:
    last = _last_rows(runs)
    if last is not None and last > day:
        raise RunError(
            f"the slots of {run_key} already hold rows through {last}; promote on {last} or later"
        )


def _failed(checked: list[SlotGates]) -> list[dict[str, Any]]:
    return [
        {
            "slot": str(result.slot),
            "run_id": result.run_id,
            "gates": [
                {
                    "name": gate.name,
                    "reason": gate.reason,
                    "value": gate.value,
                    "threshold": gate.threshold,
                    "sample": gate.sample,
                    "sample_unit": SAMPLE_UNITS[gate.name],
                }
                for gate in result.gates
                if gate.passed is False
            ],
        }
        for result in checked
        if not result.passed
    ]


def _note(
    day: date,
    scope: str,
    run_key: str,
    failed: list[dict[str, Any]],
    reason: str,
    promoted: bool,
) -> DataQualityLog:
    """The row of one promotion: info when every gate passed, else a warning to send."""
    detail = {
        "scope": scope,
        "run_key": run_key,
        "forced": bool(failed and reason),
        "reason": (reason or None) if failed else None,
        "failed": failed,
        "promoted": promoted,
    }
    severity = DataQualityLog.Severity.WARNING if failed else DataQualityLog.Severity.INFO
    return DataQualityLog.objects.create(
        date=day, check_name=PROMOTION_CHECK, severity=severity, detail=detail
    )


def _slot_runs(slot: Slot, statuses: Iterable[str]) -> list[ModelRun]:
    runs = ModelRun.objects.filter(
        objective=slot.objective, params__horizon=slot.horizon, status__in=list(statuses)
    ).exclude(model_type=SCREENING_MODEL)
    return list(runs.order_by("run_id"))


def _swap(
    runs: dict[Slot, ModelRun],
    checked: list[SlotGates],
    day: date,
    failed: list[dict[str, Any]],
    reason: str,
) -> list[str]:
    """Retire every serving run of N's slots and make N active from day, noting how."""
    judged = {result.slot: result.serving for result in checked}
    retired = []
    for slot, run in runs.items():
        old = serving_run(slot)
        if (None if old is None else old.run_id) != judged[slot]:
            raise RunError(
                f"the serving run of {slot} changed since its gates were checked; promote again"
            )
        serving = _slot_runs(slot, SERVING)
        ModelRun.objects.filter(pk__in=[item.pk for item in serving]).update(
            status=ModelStatus.RETIRED
        )
        retired += [item.run_id for item in serving]
        current = ModelRun.objects.get(pk=run.pk)
        own = [item for item in failed if item["run_id"] == run.run_id]
        note = {"forced": bool(own), "reason": (reason or None) if failed else None, "failed": own}
        params = {**current.params, PROMOTED_ON: day.isoformat(), PROMOTION: note}
        ModelRun.objects.filter(pk=run.pk).update(status=ModelStatus.ACTIVE, params=params)
    return sorted(retired)


@locked("promote")
def promote(
    run_key: str, day: date, directory: Path, force: bool = False, reason: str = ""
) -> Promoted:
    """Swap build run_key in from day when every gate passes, or by force with a reason."""
    reason = reason.strip()
    if force and not reason:
        raise RunError("--force needs a --reason saying why the failed gates are overridden")
    if reason and not force:
        raise RunError("--reason goes with --force only")
    scope = _scope(run_key)
    granted = production_pass(scope)
    _trading(day)
    with transaction.atomic():
        runs = _check_build(run_key, day, granted.frozen_by)
        checked = check_gates(run_key, directory)
        failed = _failed(checked)
        if failed and not force:
            note = _note(day, scope, run_key, failed, "", promoted=False)
            result = Promoted(False, checked, failed=failed)
        else:
            retired = _swap(runs, checked, day, failed, reason)
            _refuse_later_rows(run_key, runs, day)
            written = predict(day, directory, replace=True, slots=list(runs)).written
            note = _note(day, scope, run_key, failed, reason, promoted=True)
            result = Promoted(True, checked, retired, written, failed)
    if note.severity == DataQualityLog.Severity.WARNING:
        dispatch([note])
    if result.promoted:
        recheck_outputs(day)
    return result


def _active_in(slot: Slot, run_id: str) -> list[str]:
    runs = _slot_runs(slot, [ModelStatus.ACTIVE])
    return [run.run_id for run in runs if run.run_id != run_id]


@locked("restore")
def restore(run_id: str, day: date, reason: str) -> ModelRun:
    """Put a degraded run back to active from day; its degraded days become a closed period."""
    reason = reason.strip()
    if not reason:
        raise RunError("restoring a run needs a reason")
    if day > timezone.localdate():
        raise RunError(f"{day} is after today; a run is restored from today or an earlier day")
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
        served = Prediction.objects.filter(run=run).aggregate(last=Max("date"))["last"]
        if served is not None and day < served:
            raise RunError(
                f"{run_id} served degraded rows through {served}; restore it from then or later"
            )
        slot = Slot(run.objective, int(run.params["horizon"]))
        active = _active_in(slot, run_id)
        if active:
            raise RunError(f"{', '.join(active)} is active for {slot}; retire it first")
        periods = run.params.get(ALERT_PERIODS, [])
        if not isinstance(periods, list):
            raise RunError(f"{run_id} alert_periods is not a list; fix it before restoring")
        entry = {"start": since.isoformat(), "end": day.isoformat(), "reason": reason}
        dropped = (DEGRADED_ON, DEGRADED_WINDOW)
        kept = {name: value for name, value in run.params.items() if name not in dropped}
        run.status = ModelStatus.ACTIVE
        run.params = {**kept, ALERT_PERIODS: [*periods, entry], RESTORED_ON: day.isoformat()}
        run.save(update_fields=["status", "params"])
    return run
