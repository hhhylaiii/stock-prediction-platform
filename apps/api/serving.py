"""Which model runs serve a prediction date, and the run each displayed field is read from."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from statistics import fmean
from typing import Any

from django.db.models import Max, Q, QuerySet

from apps.research.models import ModelMonitor, ModelObjective, ModelStatus, Prediction
from apps.research.runs import SCREENING_MODEL

STATUS_ORDER = (ModelStatus.ACTIVE, ModelStatus.DEGRADED, ModelStatus.RETIRED)
SERVING = (ModelStatus.ACTIVE, ModelStatus.DEGRADED)
DEFAULT_HORIZON = 20
MONITOR_DAYS = 60
MONITOR_DIGITS = 4
RUN_FIELDS = ("model_type", "version", "status", "train_start", "train_end", "created_at")
SERVING_FIELDS = ("run_id", "run__objective", "horizon", *(f"run__{name}" for name in RUN_FIELDS))


@dataclass(frozen=True)
class Slot:
    objective: str
    horizon: int


@dataclass(frozen=True)
class ServingRun:
    run_id: str
    objective: str
    horizon: int
    model_type: str
    version: str
    status: str
    train_start: date
    train_end: date
    created_at: datetime

    def order(self) -> tuple[int, float, str]:
        return (STATUS_ORDER.index(self.status), -self.created_at.timestamp(), self.run_id)


def roles_for(horizon: int) -> dict[str, Slot]:
    """Where each displayed field comes from; the 20-day return is the quantile model's q50."""
    roles = {"rank": Slot(ModelObjective.RANK, horizon)}
    if horizon == 20:
        roles["return"] = Slot(ModelObjective.QUANTILE, 20)
        roles["interval"] = Slot(ModelObjective.QUANTILE, 20)
    else:
        roles["return"] = Slot(ModelObjective.REGRESSION, horizon)
    roles["volatility"] = Slot(ModelObjective.VOLATILITY, 20)
    if horizon == 20:
        roles["probability"] = Slot(ModelObjective.CLASSIFICATION, 20)
    return roles


def eligible() -> QuerySet[Prediction]:
    """Prediction rows of runs that ever served: never candidates, never the feature screening."""
    return Prediction.objects.exclude(run__status=ModelStatus.CANDIDATE).exclude(
        run__model_type=SCREENING_MODEL
    )


def latest_served_date(horizon: int) -> date | None:
    rows = eligible().filter(
        horizon=horizon, run__objective=ModelObjective.RANK, run__status__in=SERVING
    )
    latest: date | None = rows.aggregate(latest=Max("date"))["latest"]
    return latest


def best_runs(candidates: Iterable[ServingRun], slots: Iterable[Slot]) -> dict[Slot, ServingRun]:
    """Per slot: active before degraded before retired, then the newest run."""
    wanted = set(slots)
    best: dict[Slot, ServingRun] = {}
    for run in candidates:
        slot = Slot(run.objective, run.horizon)
        if slot in wanted and (slot not in best or run.order() < best[slot].order()):
            best[slot] = run
    return best


def runs_on(day: date, slots: Iterable[Slot]) -> dict[Slot, ServingRun]:
    """The run whose rows count for each slot on the day."""
    wanted = list(slots)
    condition = Q()
    for slot in wanted:
        condition |= Q(run__objective=slot.objective, horizon=slot.horizon)
    rows = eligible().filter(condition, date=day).values(*SERVING_FIELDS).distinct()
    return best_runs((serving_run(row) for row in rows), wanted)


def serving_run(row: dict[str, Any]) -> ServingRun:
    return ServingRun(
        run_id=row["run_id"],
        objective=row["run__objective"],
        horizon=row["horizon"],
        **{name: row[f"run__{name}"] for name in RUN_FIELDS},
    )


def ranks_on(day: date, horizon: int, stock_ids: list[str]) -> dict[str, int | None]:
    """Each stock's rank that day from the rank run that served the day."""
    slot = Slot(ModelObjective.RANK, horizon)
    run = runs_on(day, [slot]).get(slot)
    if run is None:
        return {}
    rows = Prediction.objects.filter(
        run_id=run.run_id, date=day, horizon=horizon, stock_id__in=stock_ids
    ).values_list("stock_id", "rank")
    return dict(rows)


def recent_mean(
    run: ServingRun | None, day: date | None, horizon: int, field: str
) -> tuple[float | None, int]:
    """Mean of a model_monitor column over the run's last 60 evaluated dates up to the day."""
    if run is None or day is None:
        return None, 0
    values = list(
        ModelMonitor.objects.filter(run_id=run.run_id, horizon=horizon, date__lte=day)
        .exclude(**{f"{field}__isnull": True})
        .order_by("-date")
        .values_list(field, flat=True)[:MONITOR_DAYS]
    )
    if not values:
        return None, 0
    return round(fmean(values), MONITOR_DIGITS), len(values)
