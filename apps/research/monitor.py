"""Post-launch monitoring: matured-prediction metrics, output consistency, rolling alerts (§7)."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from statistics import fmean
from typing import Any

import pandas as pd
from django.db import transaction

from apps.api.serving import MONITOR_LOOKBACK_MONTHS, Slot, months_before, served_monitor
from apps.etl.alerts import dispatch
from apps.etl.models import DataQualityLog
from apps.market.calendar import OutsideCalendarError, TradingDays
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

WIDTH_CHECK = "output_width_ratio"
ODDS_CHECK = "output_odds_vs_q50"
UNSCORED_CHECK = "monitor_unscored"
ALERT_CHECK = "model_alert"
CHECKS = (WIDTH_CHECK, ODDS_CHECK, UNSCORED_CHECK, ALERT_CHECK)
IC_THRESHOLD = "rolling_ic_threshold"
BRIER_BASELINE = "brier_baseline"
DEGRADED_ON = "degraded_on"
WINDOW = 60
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


def recent_values(objective: str, horizon: int, metric: str, day: date) -> list[float]:
    """The slot's last 60 out-of-sample values up to the day, chained over runs like the API."""
    slot = Slot(objective, horizon)
    window = {slot: (months_before(day, MONITOR_LOOKBACK_MONTHS), day)}
    rows = served_monitor(window).get(slot, [])
    return [row[metric] for row in reversed(rows) if row[metric] is not None][:WINDOW]


def _serving(objective: str) -> ModelRun | None:
    """The watched run serving an objective now: active before degraded, then the newest."""
    runs = ModelRun.objects.filter(objective=objective, status__in=WATCHED).exclude(
        model_type=SCREENING_MODEL
    )
    return min(
        runs,
        key=lambda run: (WATCHED.index(run.status), -run.created_at.timestamp(), run.run_id),
        default=None,
    )


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


def run_alerts(matured: Mapping[int, date]) -> list[dict[str, Any]]:
    """Entries of serving runs whose slot's 60-day mean breaches its rule or lacks its reference."""
    found = []
    for objective, rule in RULES.items():
        run = _serving(objective)
        if run is None:
            continue
        for horizon in rule.horizons:
            if horizon not in matured:
                continue
            values = recent_values(objective, horizon, rule.metric, matured[horizon])
            entry = _breach(run, rule, horizon, values) if len(values) == WINDOW else None
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
    """Move breached active runs to degraded, noting the day in params."""
    breached = {row["run_id"] for row in entries if row["reason"] != "missing_reference"}
    active = ModelRun.objects.filter(run_id__in=breached, status=ModelStatus.ACTIVE)
    runs = list(active.order_by("run_id"))
    for run in runs:
        run.status = ModelStatus.DEGRADED
        run.params = {**run.params, DEGRADED_ON: day.isoformat()}
        run.save(update_fields=["status", "params"])
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
        entries = run_alerts(matured)
        logs = [*consistency_checks(day), *_unscored_row(day, missed), *_alert_row(day, entries)]
        DataQualityLog.objects.filter(date=day, check_name__in=CHECKS).delete()
        DataQualityLog.objects.bulk_create(logs)
        degraded = _degrade(day, entries)
    dispatch([row for row in logs if _alerting(row)])
    return MonitorResult(scored, tuple(logs), tuple(degraded))
