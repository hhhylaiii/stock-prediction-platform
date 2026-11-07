"""Post-launch monitoring: matured-prediction metrics, output consistency, rolling alerts (§7)."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from statistics import fmean
from typing import Any

import pandas as pd
from django.db import transaction
from django.db.models import Min
from django.utils import timezone

from apps.api.serving import (
    ISO_DAY,
    MONITOR_DAYS,
    MONITOR_LOOKBACK_MONTHS,
    Slot,
    day_of,
    months_before,
    runs_on,
    served_monitor,
)
from apps.etl.alerts import dispatch
from apps.etl.models import DataQualityLog
from apps.market.calendar import CalendarError, OutsideCalendarError, TradingDays
from apps.research.baselines import INTERVAL_HORIZON
from apps.research.evaluation import ALERT_SHARE, odds_against_q50, wide_bands
from apps.research.metrics.forecast import (
    daily_absolute_error,
    daily_auc,
    daily_brier,
    daily_coverage,
    daily_direction_hits,
    daily_width,
)
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import (
    OUTPUT_HORIZONS,
    LabelDaily,
    ModelMonitor,
    ModelObjective,
    ModelRun,
    ModelStatus,
    Prediction,
)
from apps.research.runs import SCREENING_MODEL
from apps.research.walkforward import oos_start

WIDTH_CHECK = "output_width_ratio"
ODDS_CHECK = "output_odds_vs_q50"
UNSCORED_CHECK = "monitor_unscored"
ALERT_CHECK = "model_alert"
CHECKS = (WIDTH_CHECK, ODDS_CHECK, UNSCORED_CHECK, ALERT_CHECK)
IC_THRESHOLD = "rolling_ic_threshold"
BRIER_BASELINE = "brier_baseline"
DEGRADED_ON = "degraded_on"
DEGRADED_WINDOW = "degraded_window"
PROMOTED_ON = "promoted_on"
RESTORED_ON = "restored_on"
WINDOW = MONITOR_DAYS
GRACE_DAYS = 5
JUDGED, GRACE, RESET = "judged", "grace", "reset"
MODES = (JUDGED, GRACE, RESET)
NUMBER = (int, float)
ALERT_FIELDS: Mapping[str, tuple[type, ...]] = {
    "run_id": (str,),
    "objective": (str,),
    "horizon": (int,),
    "metric": (str,),
    "reason": (str,),
    "value": NUMBER,
    "threshold": (*NUMBER, list, type(None)),
    "days": (int,),
    "mode": (str,),
    "own_rows": (int,),
    "mostly_predecessors": (bool,),
    "grace_start": (str, type(None)),
    "grace_end": (str, type(None)),
    "restored_on": (str, type(None)),
    "reset_rows": (int, type(None)),
    "reset_needed": (int, type(None)),
}
MODE_FIELDS: Mapping[str, tuple[str, ...]] = {
    JUDGED: (),
    GRACE: ("grace_start", "grace_end"),
    RESET: ("restored_on", "reset_rows", "reset_needed"),
}
DAY_FIELDS = ("grace_start", "grace_end", "restored_on")
COVERAGE_BOUNDS = (0.70, 0.90)
OUTPUTS = ["score", "ret_pred", "q10", "q50", "q90", "vol_pred", "prob_up_5pct"]
LABELS = ["fwd_rank_pct", "fwd_ret_open", "fwd_vol_20d", "fwd_up_5pct"]
METRICS = ["ic", "coverage", "brier", "direction_acc", "mae", "width", "auc"]
TARGET: Mapping[str, str] = {
    ModelObjective.RANK: "fwd_rank_pct",
    ModelObjective.REGRESSION: "fwd_ret_open",
    ModelObjective.QUANTILE: "fwd_ret_open",
    ModelObjective.VOLATILITY: "fwd_vol_20d",
    ModelObjective.CLASSIFICATION: "fwd_up_5pct",
}
SERVED = (ModelStatus.ACTIVE, ModelStatus.DEGRADED, ModelStatus.RETIRED)
WATCHED = SERVED[:2]
BANDS: Mapping[str, list[str]] = {
    ModelObjective.QUANTILE: ["q10", "q50", "q90"],
    ModelObjective.VOLATILITY: ["vol_pred"],
    ModelObjective.CLASSIFICATION: ["prob_up_5pct"],
}
Scores = dict[str, float | None]


logger = logging.getLogger(__name__)


class MonitorError(ValueError):
    pass


def _first(series: pd.Series) -> float | None:
    return float(series.iloc[0]) if len(series) else None


def _returns(rows: pd.DataFrame, predicted: str) -> Scores:
    return {
        "mae": _first(daily_absolute_error(rows, "fwd_ret_open", predicted)),
        "direction_acc": _first(daily_direction_hits(rows, "fwd_ret_open", predicted)),
    }


SCORERS: Mapping[str, Callable[[pd.DataFrame], Scores]] = {
    ModelObjective.RANK: lambda rows: {"ic": _first(daily_rank_ic(rows, "score"))},
    ModelObjective.REGRESSION: lambda rows: _returns(rows, "ret_pred"),
    ModelObjective.QUANTILE: lambda rows: {
        "coverage": _first(daily_coverage(rows, "fwd_ret_open", "q10", "q90")),
        "width": _first(daily_width(rows, "q10", "q90")),
        **_returns(rows, "q50"),
    },
    ModelObjective.VOLATILITY: lambda rows: {
        "mae": _first(daily_absolute_error(rows, "fwd_vol_20d", "vol_pred"))
    },
    ModelObjective.CLASSIFICATION: lambda rows: {
        "brier": _first(daily_brier(rows, "fwd_up_5pct", "prob_up_5pct")),
        "auc": _first(daily_auc(rows, "fwd_up_5pct", "prob_up_5pct")),
    },
}


def matured_dates(calendar: TradingDays, day: date) -> dict[int, date]:
    """Per horizon, the prediction date whose t+h close is the day, counted in trading days."""
    found = {}
    for horizon in OUTPUT_HORIZONS:
        try:
            found[horizon] = calendar.shift(day, -horizon)
        except OutsideCalendarError:
            continue
    return found


def _matured_rows(day: date, horizon: int) -> pd.DataFrame:
    """Predictions of a date and horizon joined to their labels."""
    predictions = Prediction.objects.filter(date=day, horizon=horizon)
    fields = ["run_id", "run__objective", "stock_id", "in_universe", *OUTPUTS]
    frame = pd.DataFrame(list(predictions.values_list(*fields)), columns=fields)
    fields = ["stock_id", "untradable_t1", *LABELS]
    labels = LabelDaily.objects.filter(date=day, horizon=horizon).values_list(*fields)
    known = pd.DataFrame(list(labels), columns=fields)
    known["fwd_up_5pct"] = known["fwd_up_5pct"].map({True: 1.0, False: 0.0})
    joined = frame.merge(known, on="stock_id", how="left")
    numbers = joined[[*OUTPUTS, *LABELS]].astype(float)
    return joined.drop(columns=[*OUTPUTS, *LABELS]).join(numbers).assign(date=day)


def score_matured(day: date, horizon: int) -> tuple[int, list[str]]:
    """Rewrite model_monitor of runs predicting the date; return the count and runs left unscored.

    Rows count as in research: universe, tradable at the next open, label known.
    """
    rows = _matured_rows(day, horizon)
    found, unscored = [], []
    for (run_id, objective), part in rows.groupby(["run_id", "run__objective"]):
        counted = part["in_universe"].astype(bool) & part["untradable_t1"].eq(False)
        kept = part[counted & part[TARGET[str(objective)]].notna()]
        if kept.empty:
            unscored.append(str(run_id))
            continue
        scores = {**dict.fromkeys(METRICS), **SCORERS[str(objective)](kept)}
        found.append(
            ModelMonitor(
                run_id=str(run_id), date=day, horizon=horizon, n_stocks=len(kept), **scores
            )
        )
    scored = [row.run_id for row in found]
    ModelMonitor.objects.filter(date=day, horizon=horizon, run_id__in=scored).delete()
    ModelMonitor.objects.bulk_create(found)
    return len(found), unscored


def _served_runs(day: date) -> dict[str, str]:
    """Per objective, the run the API serves the day from: active, degraded, retired, newest."""
    rows = Prediction.objects.filter(date=day, horizon=INTERVAL_HORIZON, run__status__in=SERVED)
    fields = ["run_id", "run__objective", "run__status", "run__created_at"]
    runs = sorted(
        set(rows.values_list(*fields)),
        key=lambda run: (SERVED.index(run[2]), -run[3].timestamp(), run[0]),
    )
    chosen: dict[str, str] = {}
    for run_id, objective, _, _ in runs:
        chosen.setdefault(objective, run_id)
    return chosen


def _outputs_on(day: date) -> pd.DataFrame:
    """Each universe stock's 20-day outputs that day from the one run serving each objective."""
    merged = pd.DataFrame({"stock_id": pd.Series([], dtype=object)})
    for objective, run_id in _served_runs(day).items():
        columns = BANDS.get(objective)
        if columns is None:
            continue
        rows = Prediction.objects.filter(
            run_id=run_id, date=day, horizon=INTERVAL_HORIZON, in_universe=True
        )
        part = pd.DataFrame(
            list(rows.values_list("stock_id", *columns)), columns=["stock_id", *columns]
        )
        merged = merged.merge(part, on="stock_id", how="outer")
    for columns in BANDS.values():
        for name in columns:
            if name not in merged:
                merged[name] = float("nan")
    return merged.astype({name: float for columns in BANDS.values() for name in columns})


def _consistency_row(
    day: date, check: str, stocks: pd.Series, flagged: pd.Series
) -> DataQualityLog:
    marked = sorted(stocks.loc[flagged[flagged].index].tolist())
    detail = {
        "share": float(flagged.mean()),
        "flagged": len(marked),
        "n_stocks": len(flagged),
        "stocks": marked,
    }
    severity = DataQualityLog.Severity.WARNING if marked else DataQualityLog.Severity.INFO
    return DataQualityLog(date=day, check_name=check, severity=severity, detail=detail)


def consistency_checks(day: date) -> list[DataQualityLog]:
    """Width-ratio and odds-versus-q50 rows of the day's outputs, each skipped without inputs."""
    frame = _outputs_on(day)
    checks = {
        WIDTH_CHECK: wide_bands(frame, "q10", "q90", "vol_pred"),
        ODDS_CHECK: odds_against_q50(frame, "q50", "prob_up_5pct"),
    }
    return [
        _consistency_row(day, check, frame["stock_id"], flagged)
        for check, flagged in checks.items()
        if len(flagged)
    ]


def _unscored_row(day: date, missed: list[dict[str, Any]]) -> list[DataQualityLog]:
    """One warning naming the runs whose matured predictions had no countable label."""
    if not missed:
        return []
    detail = {"runs": missed}
    severity = DataQualityLog.Severity.WARNING
    return [DataQualityLog(date=day, check_name=UNSCORED_CHECK, severity=severity, detail=detail)]


@dataclass(frozen=True)
class Rule:
    """A watched metric of one objective, the reference it is held to and when it breaches."""

    metric: str
    horizons: tuple[int, ...]
    reason: str
    limit: Callable[[Mapping[str, Any], int], Any]
    breached: Callable[[float, Any], bool]


def _reference(key: str, by_horizon: bool) -> Callable[[Mapping[str, Any], int], Any]:
    def read(params: Mapping[str, Any], horizon: int) -> Any:
        value = params.get(key)
        return value.get(str(horizon)) if by_horizon and isinstance(value, dict) else value

    return read


RULES: Mapping[str, Rule] = {
    ModelObjective.RANK: Rule(
        "ic",
        tuple(OUTPUT_HORIZONS),
        "below_threshold",
        _reference(IC_THRESHOLD, by_horizon=True),
        lambda value, limit: value < limit,
    ),
    ModelObjective.QUANTILE: Rule(
        "coverage",
        (INTERVAL_HORIZON,),
        "outside_range",
        lambda params, horizon: list(COVERAGE_BOUNDS),
        lambda value, limit: not limit[0] <= value <= limit[1],
    ),
    ModelObjective.CLASSIFICATION: Rule(
        "brier",
        (INTERVAL_HORIZON,),
        "above_baseline",
        _reference(BRIER_BASELINE, by_horizon=False),
        lambda value, limit: value > limit,
    ),
}


def recent_rows(objective: str, horizon: int, metric: str, day: date) -> list[dict[str, Any]]:
    """The slot's last 60 out-of-sample rows with the metric up to the day, newest first."""
    slot = Slot(objective, horizon)
    window = {slot: (months_before(day, MONITOR_LOOKBACK_MONTHS), day)}
    rows = served_monitor(window).get(slot, [])
    return [row for row in reversed(rows) if row[metric] is not None][:WINDOW]


def _breach(run: ModelRun, rule: Rule, horizon: int, values: list[float]) -> dict[str, Any] | None:
    """The run's entry in the day's alert when its mean breaches or its reference is missing."""
    value, threshold = fmean(values), rule.limit(run.params, horizon)
    if threshold is not None and not rule.breached(value, threshold):
        return None
    return {
        "run_id": run.run_id,
        "objective": run.objective,
        "horizon": horizon,
        "metric": rule.metric,
        "reason": rule.reason if threshold is not None else "missing_reference",
        "value": value,
        "threshold": threshold,
        "days": len(values),
    }


def _type_problem(name: str, value: Any) -> str | None:
    allowed = ALERT_FIELDS[name]
    if isinstance(value, bool) and bool not in allowed:
        return f"{name} is a bool, not one of {[kind.__name__ for kind in allowed]}"
    if not isinstance(value, allowed):
        return f"{name} is {type(value).__name__}, not one of {[kind.__name__ for kind in allowed]}"
    if name in DAY_FIELDS and isinstance(value, str) and not ISO_DAY.fullmatch(value):
        return f"{name} {value!r} is not a YYYY-MM-DD day"
    return None


def alert_problems(entry: Mapping[str, Any]) -> list[str]:
    """What keeps a model_alert entry off the contract; empty when the entry is on it."""
    problems = [f"{name} is missing" for name in ALERT_FIELDS if name not in entry]
    problems += [f"{name} is not a field" for name in entry if name not in ALERT_FIELDS]
    problems += [
        found
        for name in ALERT_FIELDS
        if name in entry and (found := _type_problem(name, entry[name])) is not None
    ]
    mode = entry.get("mode")
    if not isinstance(mode, str) or mode not in MODE_FIELDS:
        return [*problems, f"mode {mode!r} is not one of {list(MODES)}"]
    filled = MODE_FIELDS[mode]
    for name in [name for names in MODE_FIELDS.values() for name in names]:
        if name in filled and entry.get(name) is None:
            problems.append(f"{name} is null in mode {mode}")
        if name not in filled and entry.get(name) is not None:
            problems.append(f"{name} is set outside its mode, in mode {mode}")
    return problems


def _served_on(day: date, slot: Slot) -> ModelRun | None:
    """The run the API serves the slot from on the day, locked until the day is written."""
    found = runs_on(day, [slot]).get(slot)
    if found is None:
        return None
    return ModelRun.objects.select_for_update().get(pk=found.run_id)


def _first_session(calendar: TradingDays, day: date) -> date:
    return (
        day
        if calendar.is_trading_day(day)
        else calendar.shift(calendar.latest_on_or_before(day), 1)
    )


def grace_start(run: ModelRun, horizon: int, calendar: TradingDays) -> date:
    """max(oos_start, promoted_on); runs promoted before promoted_on existed use their first day."""
    boundary = oos_start(calendar, run.train_end, horizon)
    promoted = day_of(run.params.get(PROMOTED_ON))
    if promoted is None:
        created = _first_session(calendar, timezone.localtime(run.created_at).date())
        first = run.predictions.filter(horizon=horizon).aggregate(first=Min("date"))["first"]
        promoted = created if first is None else max(first, created)
    return max(boundary, promoted)


def _restored_on(judged: ModelRun, slot: Slot, day: date) -> date | None:
    """The slot's latest restore: the judged run's own at any date, others' on or before day."""
    runs = (
        ModelRun.objects.filter(objective=slot.objective, params__has_key=RESTORED_ON)
        .exclude(status=ModelStatus.CANDIDATE)
        .exclude(model_type=SCREENING_MODEL)
        .exclude(pk=judged.pk)
    )
    days = [
        restored
        for run in runs
        if (restored := day_of(run.params.get(RESTORED_ON))) is not None
        and restored <= day
        and run.predictions.filter(horizon=slot.horizon).exists()
    ]
    own = day_of(judged.params.get(RESTORED_ON))
    return max([*days, *([] if own is None else [own])], default=None)


def _mode(
    run: ModelRun, slot: Slot, rows: list[dict[str, Any]], day: date, calendar: TradingDays
) -> tuple[str, dict[str, Any]]:
    """judged, or grace or reset with their fields; a reset comes before a grace."""
    restored = _restored_on(run, slot, day)
    if restored is not None:
        since = sum(1 for row in rows if row["date"] >= restored)
        if since < WINDOW:
            extra = {"restored_on": restored.isoformat(), "reset_rows": since}
            return RESET, {**extra, "reset_needed": WINDOW}
    start = _first_session(calendar, grace_start(run, slot.horizon, calendar))
    spared = calendar.between(start, calendar.end)[:GRACE_DAYS]
    if len(spared) < GRACE_DAYS:
        logger.warning(
            "the calendar ends inside the grace of %s from %s; extend it", run.run_id, start
        )
    if day < start or (spared and day <= spared[-1]):
        last = spared[-1] if spared else start
        return GRACE, {"grace_start": start.isoformat(), "grace_end": last.isoformat()}
    return JUDGED, {}


def _entry(
    run: ModelRun,
    rule: Rule,
    slot: Slot,
    rows: list[dict[str, Any]],
    mode: str,
    extra: dict[str, Any],
) -> dict[str, Any] | None:
    found = _breach(run, rule, slot.horizon, [row[rule.metric] for row in rows])
    if found is None:
        return None
    own = sum(1 for row in rows if row["run_id"] == run.run_id)
    empty = dict.fromkeys(name for names in MODE_FIELDS.values() for name in names)
    marks = {"mode": mode, "own_rows": own, "mostly_predecessors": own * 2 < len(rows)}
    return {**found, **marks, **empty, **extra}


def run_alerts(
    day: date, matured: Mapping[int, date], calendar: TradingDays
) -> list[dict[str, Any]]:
    """Per slot, the entry of the run serving the day when its slot's 60-row window breaches."""
    watched = ModelRun.objects.filter(status__in=WATCHED, objective__in=list(RULES))
    list(watched.select_for_update().order_by("run_id"))
    found = []
    for objective, rule in RULES.items():
        for horizon in rule.horizons:
            if horizon not in matured:
                continue
            slot = Slot(objective, horizon)
            run = _served_on(day, slot)
            if run is None or run.status not in WATCHED:
                continue
            rows = recent_rows(objective, horizon, rule.metric, matured[horizon])
            if len(rows) < WINDOW:
                continue
            try:
                mode, extra = _mode(run, slot, rows, day, calendar)
            except CalendarError as error:
                logger.warning(
                    "run %s is not judged on %s for %s: %s", run.run_id, day, slot, error
                )
                continue
            entry = _entry(run, rule, slot, rows, mode, extra)
            if entry is not None:
                found.append(entry)
    return found


def _alert_row(day: date, entries: list[dict[str, Any]]) -> list[DataQualityLog]:
    """One warning row listing every alerted run of the day."""
    if not entries:
        return []
    severity = DataQualityLog.Severity.WARNING
    detail = {"runs": entries}
    return [DataQualityLog(date=day, check_name=ALERT_CHECK, severity=severity, detail=detail)]


def _degrade(day: date, entries: list[dict[str, Any]]) -> list[str]:
    """Move judged, breached active runs to degraded under a row lock, noting day and window."""
    breached: dict[str, dict[str, Any]] = {}
    for row in entries:
        if row["mode"] == JUDGED and row["reason"] != "missing_reference":
            breached.setdefault(row["run_id"], row)
    locked = ModelRun.objects.select_for_update().filter(run_id__in=breached)
    runs = [run for run in locked.order_by("run_id") if run.status == ModelStatus.ACTIVE]
    for run in runs:
        entry = breached[run.run_id]
        window = {name: entry[name] for name in ("own_rows", "days", "mostly_predecessors")}
        params = {**run.params, DEGRADED_ON: day.isoformat(), DEGRADED_WINDOW: window}
        ModelRun.objects.filter(pk=run.pk).update(status=ModelStatus.DEGRADED, params=params)
    return [run.run_id for run in runs]


def _alerting(row: DataQualityLog) -> bool:
    if row.check_name in (UNSCORED_CHECK, ALERT_CHECK):
        return True
    return bool(row.detail["share"] > ALERT_SHARE)


@dataclass(frozen=True)
class MonitorResult:
    scored: int
    logs: tuple[DataQualityLog, ...]
    degraded: tuple[str, ...]


def run_monitor(day: date, calendar: TradingDays) -> MonitorResult:
    """Score predictions maturing on the day, check its outputs, raise alerts; reruns rewrite."""
    if not calendar.is_trading_day(day):
        raise MonitorError(f"{day} is not a trading day")
    matured = matured_dates(calendar, day)
    with transaction.atomic():
        scored, missed = 0, []
        for horizon, when in matured.items():
            count, unscored = score_matured(when, horizon)
            scored += count
            missed += [
                {"run_id": run_id, "date": when.isoformat(), "horizon": horizon}
                for run_id in unscored
            ]
        entries = run_alerts(day, matured, calendar)
        logs = [*consistency_checks(day), *_unscored_row(day, missed), *_alert_row(day, entries)]
        DataQualityLog.objects.filter(date=day, check_name__in=CHECKS).delete()
        DataQualityLog.objects.bulk_create(logs)
        degraded = _degrade(day, entries)
    dispatch([row for row in logs if _alerting(row)])
    return MonitorResult(scored, tuple(logs), tuple(degraded))
