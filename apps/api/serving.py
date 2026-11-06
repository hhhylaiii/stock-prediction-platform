"""Which model runs serve a prediction date, and the run each displayed field is read from."""

import calendar
import functools
import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from statistics import fmean
from typing import Any

from django.db.models import F, Max, Q, QuerySet

from apps.research.models import ModelMonitor, ModelObjective, ModelStatus, Prediction
from apps.research.runs import SCREENING_MODEL

STATUS_ORDER = (ModelStatus.ACTIVE, ModelStatus.DEGRADED, ModelStatus.RETIRED)
SERVING = (ModelStatus.ACTIVE, ModelStatus.DEGRADED)
DEFAULT_HORIZON = 20
MONITOR_DAYS = 60
MONITOR_DIGITS = 4
MONITOR_LOOKBACK_MONTHS = 6
MONITOR_FIELDS = ("ic", "coverage", "width", "brier", "auc", "direction_acc", "mae")
RUN_FIELDS = (
    "run_key",
    "model_type",
    "version",
    "status",
    "train_start",
    "train_end",
    "created_at",
)
DEGRADED_ON = "run__params__degraded_on"
ALERT_PERIODS = "run__params__alert_periods"
ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
SERVING_FIELDS = (
    "run_id",
    "run__objective",
    "horizon",
    *(f"run__{name}" for name in RUN_FIELDS),
    DEGRADED_ON,
    ALERT_PERIODS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Slot:
    objective: str
    horizon: int


@dataclass(frozen=True)
class ServingRun:
    run_id: str
    objective: str
    horizon: int
    run_key: str
    model_type: str
    version: str
    status: str
    train_start: date
    train_end: date
    created_at: datetime
    degraded_on: date | None = None
    alert_periods: tuple[tuple[date, date], ...] = ()

    def order(self) -> tuple[int, float, str]:
        return (STATUS_ORDER.index(self.status), -self.created_at.timestamp(), self.run_id)


@functools.cache
def warn_undated(run_id: str) -> None:
    logger.warning(
        "degraded run %s has no readable params.degraded_on; its band is set aside on every day",
        run_id,
    )


@functools.cache
def warn_unreadable_periods(run_id: str) -> None:
    logger.warning("run %s has unreadable params.alert_periods entries; they are left out", run_id)


def day_of(value: object) -> date | None:
    """A params date, or None unless it is a YYYY-MM-DD string."""
    if not isinstance(value, str) or not ISO_DAY.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def periods_of(run_id: str, value: object) -> tuple[tuple[date, date], ...]:
    """The readable [start, end) periods of params.alert_periods; a warning names the rest."""
    if value is None:
        return ()
    if not isinstance(value, list):
        warn_unreadable_periods(run_id)
        return ()
    periods = []
    for entry in value:
        if isinstance(entry, dict):
            start, end = day_of(entry.get("start")), day_of(entry.get("end"))
            if start is not None and end is not None and start <= end:
                periods.append((start, end))
    if len(periods) < len(value):
        warn_unreadable_periods(run_id)
    return tuple(periods)


def interval_alert(run: ServingRun | None, day: date) -> bool:
    """The volatility band stands in inside a closed alert period or from an open degraded_on."""
    if run is None:
        return False
    if any(start <= day < end for start, end in run.alert_periods):
        return True
    if run.status == ModelStatus.ACTIVE:
        return False
    if run.degraded_on is not None:
        return day >= run.degraded_on
    if run.status == ModelStatus.DEGRADED:
        warn_undated(run.run_id)
        return True
    return False


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
        degraded_on=day_of(row.get(DEGRADED_ON)),
        alert_periods=periods_of(row["run_id"], row.get(ALERT_PERIODS)),
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


def months_before(day: date, count: int) -> date:
    """The same day count months earlier, clamped to that month's last day."""
    year, month = divmod(day.year * 12 + day.month - 1 - count, 12)
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


def served_monitor(windows: dict[Slot, tuple[date, date]]) -> dict[Slot, list[dict[str, Any]]]:
    """Per slot and day in (after, through], the monitor row of the run that served it, OOS only."""
    if not windows:
        return {}
    condition = Q()
    for slot, (after, through) in windows.items():
        condition |= Q(
            run__objective=slot.objective, horizon=slot.horizon, date__gt=after, date__lte=through
        )
    rows = (
        ModelMonitor.objects.filter(condition, date__gt=F("run__train_end"))
        .exclude(run__status=ModelStatus.CANDIDATE)
        .exclude(run__model_type=SCREENING_MODEL)
        .values("date", *MONITOR_FIELDS, *SERVING_FIELDS)
    )
    by_day: dict[tuple[Slot, date], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[(Slot(row["run__objective"], row["horizon"]), row["date"])].append(row)
    served: dict[Slot, list[dict[str, Any]]] = defaultdict(list)
    for (slot, _), candidates in sorted(by_day.items(), key=lambda item: item[0][1]):
        best = best_runs((serving_run(row) for row in candidates), [slot])[slot]
        served[slot].append(next(row for row in candidates if row["run_id"] == best.run_id))
    return served


def recent_mean(run: ServingRun | None, day: date | None, field: str) -> tuple[float | None, int]:
    """Mean of a monitor column over the slot's last 60 evaluated days up to the day."""
    if run is None or day is None:
        return None, 0
    slot = Slot(run.objective, run.horizon)
    window = {slot: (months_before(day, MONITOR_LOOKBACK_MONTHS), day)}
    rows = served_monitor(window).get(slot, [])
    values = [row[field] for row in reversed(rows) if row[field] is not None][:MONITOR_DAYS]
    if not values:
        return None, 0
    return round(fmean(values), MONITOR_DIGITS), len(values)
