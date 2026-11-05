"""References the monitor holds runs to, from walk-forward research folds (SPEC §7)."""

import math
from collections.abc import Iterable, Mapping

import pandas as pd

from apps.research.baseline_runs import BASELINE
from apps.research.labels import VOL_DAYS
from apps.research.metrics.forecast import daily_brier
from apps.research.metrics.ic import daily_rank_ic, research_ic
from apps.research.models import OUTPUT_HORIZONS, ModelRun
from apps.research.monitor import BRIER_BASELINE, IC_THRESHOLD, WINDOW
from apps.research.prediction_cache import load_predictions
from apps.research.training import load_labels
from apps.research.walkforward import Fold

QUANTILE = 0.05
LOGISTIC = f"logistic_h{VOL_DAYS}"


def rolling_ic_threshold(ic: pd.Series, window: int = WINDOW, quantile: float = QUANTILE) -> float:
    """Low quantile of the mean IC over every run of window consecutive IC days."""
    means = ic.sort_index().rolling(window).mean().dropna()
    return float(means.quantile(quantile)) if len(means) else math.nan


def _labelled(cached: pd.DataFrame, folds: list[Fold], horizon: int) -> pd.DataFrame:
    """Cached universe rows joined to their label_daily row of a horizon."""
    first = min(fold.test_start for fold in folds)
    last = max(fold.eval_end for fold in folds)
    joined = cached.merge(load_labels(first, last, horizon), on=["date", "stock_id"])
    return joined[joined["in_universe"].astype(bool)]


def research_ic_thresholds(run_key: str, model: str, folds: Iterable[Fold]) -> dict[int, float]:
    """Per horizon, the 5% quantile of 60-day rolling rank IC over the research folds."""
    chosen = [fold for fold in folds if fold.research]
    cached = load_predictions(run_key, model, chosen)
    thresholds = {}
    for horizon in OUTPUT_HORIZONS:
        column = f"{model}_h{horizon}"
        if column in cached:
            rows = _labelled(cached, chosen, horizon)
            thresholds[horizon] = rolling_ic_threshold(
                research_ic(daily_rank_ic(rows, column), chosen)
            )
    return thresholds


def research_brier(run_key: str, folds: Iterable[Fold]) -> float:
    """Mean daily Brier of the logistic baseline over the research folds."""
    chosen = [fold for fold in folds if fold.research]
    rows = _labelled(load_predictions(run_key, BASELINE, chosen), chosen, VOL_DAYS)
    outcomes = rows.assign(fwd_up_5pct=rows["fwd_up_5pct"].map({True: 1.0, False: 0.0}))
    daily = research_ic(daily_brier(outcomes, "fwd_up_5pct", LOGISTIC), chosen)
    return float(daily.mean()) if len(daily) else math.nan


def store_references(run_id: str, thresholds: Mapping[int, float], brier: float) -> ModelRun:
    """Replace a run's IC thresholds and baseline Brier in params, leaving non-finite values out."""
    params = ModelRun.objects.get(pk=run_id).params
    kept = {
        key: value for key, value in params.items() if key not in (IC_THRESHOLD, BRIER_BASELINE)
    }
    limits = {str(h): float(v) for h, v in sorted(thresholds.items()) if math.isfinite(v)}
    brier_part = {BRIER_BASELINE: float(brier)} if math.isfinite(brier) else {}
    ModelRun.objects.filter(pk=run_id).update(params={**kept, IC_THRESHOLD: limits, **brier_part})
    return ModelRun.objects.get(pk=run_id)
