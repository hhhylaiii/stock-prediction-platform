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

from django.db.models import F, OuterRef, Q, QuerySet, Subquery

from apps.market.calendar import CalendarError, TradingDays
from apps.research.holdout import DEGRADED, FAILED, PRODUCTION
from apps.research.models import ModelMonitor, ModelObjective, ModelRun, ModelStatus, Prediction
from apps.research.runs import SCREENING_MODEL
from apps.research.walkforward import oos_start

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
HOLDOUT_RELEASE = "holdout_release"
RELEASE = f"params__{PRODUCTION}__{HOLDOUT_RELEASE}"
RELEASE_TEXTS = ("status", "reason", "released_at")
RELEASE_LISTS = ("failed", "flagged")
PROMOTION = "promotion"
PROMOTION_CHECK = "model_promotion"
GATE_FIELDS = ("name", "reason", "value", "threshold", "sample", "sample_unit")
NO_CAVEATS: dict[str, Any] = {"released_verdict": None, "forced_gates": None}
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


@functools.cache
def warn_unreadable(run_id: str, note: str) -> None:
    logger.warning("run %s has an unreadable %s; the pages leave it out", run_id, note)


def is_names(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(name, str) for name in value)


def release_of(value: object) -> dict[str, Any] | None:
    """A recorded holdout release without its report path or hash; None unless all of it reads."""
    if not isinstance(value, dict):
        return None
    shown = {name: value.get(name) for name in (*RELEASE_TEXTS, *RELEASE_LISTS)}
    texts = all(isinstance(shown[name], str) for name in RELEASE_TEXTS)
    lists = all(is_names(shown[name]) for name in RELEASE_LISTS)
    return shown if texts and lists and shown["status"] in (FAILED, DEGRADED) else None


def forced_gates(value: object) -> list[dict[str, Any]] | None:
    """The gates a forced promotion overrode; None unless the note is forced and names them."""
    if not isinstance(value, dict) or value.get("forced") is not True:
        return None
    failed = value.get("failed")
    items = failed if isinstance(failed, list) else []
    listed = [item.get("gates") for item in items if isinstance(item, dict)]
    gates = [gate for found in listed if isinstance(found, list) for gate in found]
    named = all(isinstance(gate, dict) and isinstance(gate.get("name"), str) for gate in gates)
    if not gates or not named:
        return None
    return [{name: gate.get(name) for name in GATE_FIELDS} for gate in gates]


def caveats(run_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Per run, read once: the verdict its holdout was released on and the gates forced past."""
    runs = ModelRun.objects.filter(run_id__in=set(run_ids))
    found = {}
    for run_id, recorded, promotion in runs.values_list("run_id", RELEASE, f"params__{PROMOTION}"):
        release, gates = release_of(recorded), forced_gates(promotion)
        if release is None and recorded is not None:
            warn_unreadable(run_id, "holdout release")
        if gates is None and isinstance(promotion, dict) and promotion.get("forced") is True:
            warn_unreadable(run_id, "forced promotion")
        found[run_id] = {
            "released_verdict": None if release is None else release["status"],
            "forced_gates": None if gates is None else [gate["name"] for gate in gates],
        }
    return found


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


def latest_days(slots: Iterable[Slot]) -> dict[Slot, date]:
    """Each slot's latest day among its active and degraded runs, one index probe per run."""
    wanted = set(slots)
    horizons = sorted({slot.horizon for slot in wanted})
    newest = {
        f"h{horizon}": Subquery(
            Prediction.objects.filter(run_id=OuterRef("run_id"), horizon=horizon)
            .order_by("-date")
            .values("date")[:1]
        )
        for horizon in horizons
    }
    runs = (
        ModelRun.objects.filter(
            status__in=SERVING, objective__in={slot.objective for slot in wanted}
        )
        .exclude(model_type=SCREENING_MODEL)
        .annotate(**newest)
        .values("objective", *newest)
    )
    days: dict[Slot, date] = {}
    for row in runs:
        for horizon in horizons:
            slot, day = Slot(row["objective"], horizon), row[f"h{horizon}"]
            if slot in wanted and day is not None and (slot not in days or day > days[slot]):
                days[slot] = day
    return days


def latest_served_date(horizon: int) -> date | None:
    slot = Slot(ModelObjective.RANK, horizon)
    return latest_days([slot]).get(slot)


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


def runs_on_days(days: dict[Slot, date]) -> dict[Slot, ServingRun]:
    """The run whose rows count for each slot on that slot's own day."""
    if not days:
        return {}
    condition = Q()
    for slot, day in days.items():
        condition |= Q(run__objective=slot.objective, horizon=slot.horizon, date=day)
    rows = eligible().filter(condition).values(*SERVING_FIELDS).distinct()
    return best_runs((serving_run(row) for row in rows), days)


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


@functools.cache
def warn_unplaced(run_id: str, horizon: int, reason: str) -> None:
    logger.warning(
        "run %s has no %s-day out-of-sample day in the calendar (%s); its rows are left out",
        run_id,
        horizon,
        reason,
    )


def first_oos_day(calendar: TradingDays, run_id: str, train_end: date, horizon: int) -> date | None:
    """predict's out-of-sample boundary; None, with a warning, when the calendar cannot place it."""
    try:
        return oos_start(calendar, train_end, horizon)
    except CalendarError as error:
        warn_unplaced(run_id, horizon, str(error))
        return None


def out_of_sample(rows: list[dict[str, Any]], calendar: TradingDays) -> list[dict[str, Any]]:
    """Rows on or after their run's first out-of-sample day for the row's own horizon."""
    trained = {(row["run_id"], row["horizon"]): row["run__train_end"] for row in rows}
    starts = {
        (run_id, horizon): first_oos_day(calendar, run_id, end, horizon)
        for (run_id, horizon), end in trained.items()
    }
    return [
        row
        for row in rows
        if (start := starts[(row["run_id"], row["horizon"])]) is not None and row["date"] >= start
    ]


def monitor_calendar() -> TradingDays | None:
    """The calendar for monitor rows; None, logged as an error, when it cannot be read."""
    try:
        return TradingDays.from_db()
    except CalendarError as error:
        logger.error(
            "trading calendar cannot be read (%s); no monitor row counts as out of sample", error
        )
        return None


def served_monitor(windows: dict[Slot, tuple[date, date]]) -> dict[Slot, list[dict[str, Any]]]:
    """Per slot and day in (after, through], the monitor row of the run that served it, OOS only."""
    if not windows:
        return {}
    condition = Q()
    for slot, (after, through) in windows.items():
        condition |= Q(
            run__objective=slot.objective, horizon=slot.horizon, date__gt=after, date__lte=through
        )
    rows = list(
        ModelMonitor.objects.filter(condition, date__gt=F("run__train_end"))
        .exclude(run__status=ModelStatus.CANDIDATE)
        .exclude(run__model_type=SCREENING_MODEL)
        .values("date", *MONITOR_FIELDS, *SERVING_FIELDS)
    )
    calendar = monitor_calendar() if rows else None
    by_day: dict[tuple[Slot, date], list[dict[str, Any]]] = defaultdict(list)
    for row in [] if calendar is None else out_of_sample(rows, calendar):
        by_day[(Slot(row["run__objective"], row["horizon"]), row["date"])].append(row)
    served: dict[Slot, list[dict[str, Any]]] = defaultdict(list)
    for (slot, _), candidates in sorted(by_day.items(), key=lambda item: item[0][1]):
        best = best_runs((serving_run(row) for row in candidates), [slot])[slot]
        served[slot].append(next(row for row in candidates if row["run_id"] == best.run_id))
    return served


def column_mean(rows: list[dict[str, Any]], field: str) -> tuple[float | None, int]:
    """Mean of a monitor column over the last 60 evaluated days among the rows."""
    values = [row[field] for row in reversed(rows) if row[field] is not None][:MONITOR_DAYS]
    if not values:
        return None, 0
    return round(fmean(values), MONITOR_DIGITS), len(values)


def recent_means(
    day: date | None, picks: list[tuple[ServingRun | None, str]]
) -> list[tuple[float | None, int]]:
    """Each run's column mean over its slot's last 60 evaluated days, from one monitor read."""
    slots = {Slot(run.objective, run.horizon) for run, _ in picks if run is not None}
    if day is None or not slots:
        return [(None, 0)] * len(picks)
    served = served_monitor(
        dict.fromkeys(slots, (months_before(day, MONITOR_LOOKBACK_MONTHS), day))
    )
    return [
        (None, 0)
        if run is None
        else column_mean(served.get(Slot(run.objective, run.horizon), []), field)
        for run, field in picks
    ]
