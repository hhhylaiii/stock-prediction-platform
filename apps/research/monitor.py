"""Post-launch monitoring: matured-prediction metrics and output consistency (SPEC §7)."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from django.db import transaction

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
    ModelStatus,
    Prediction,
)

WIDTH_CHECK = "output_width_ratio"
ODDS_CHECK = "output_odds_vs_q50"
UNSCORED_CHECK = "monitor_unscored"
CHECKS = (WIDTH_CHECK, ODDS_CHECK, UNSCORED_CHECK)
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


def _alerting(row: DataQualityLog) -> bool:
    return row.check_name == UNSCORED_CHECK or bool(row.detail["share"] > ALERT_SHARE)


@dataclass(frozen=True)
class MonitorResult:
    scored: int
    logs: tuple[DataQualityLog, ...]


def run_monitor(day: date, calendar: TradingDays) -> MonitorResult:
    """Score predictions maturing on the day and check its outputs; reruns rewrite both."""
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
        logs = [*consistency_checks(day), *_unscored_row(day, missed)]
        DataQualityLog.objects.filter(date=day, check_name__in=CHECKS).delete()
        DataQualityLog.objects.bulk_create(logs)
    dispatch([row for row in logs if _alerting(row)])
    return MonitorResult(scored, tuple(logs))
