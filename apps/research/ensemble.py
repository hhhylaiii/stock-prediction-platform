"""Rank-average ensemble of the Ridge and LightGBM rank scores on cached folds (SPEC §7)."""

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from apps.market.calendar import TradingDays
from apps.research import lgbm_runs
from apps.research.baseline_runs import BASELINE, ROW_KEYS
from apps.research.labels import latest_price_date
from apps.research.metrics.ic import daily_rank_ic
from apps.research.prediction_cache import fold_notes, load_predictions
from apps.research.rank_model import RANK_LABEL
from apps.research.training import load_labels, training_rows
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

ENSEMBLE = "ensemble"
MODEL = lgbm_runs.MODEL


def research_folds(config: WalkForwardConfig, last: date | None) -> list[Fold]:
    """Research folds up to last or the latest price day; tuning and holdout stay out."""
    end = last or latest_price_date()
    if end is None:
        raise ValueError("daily_price is empty")
    folds = [fold for fold in generate_folds(TradingDays.from_db(), end, config) if fold.research]
    if not folds:
        raise ValueError(f"no research folds end by {end}")
    return folds


def write_report(text: str, output: str | None) -> str | None:
    """Write a report to output when given; the path written, or None to print instead."""
    if output is None:
        return None
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def rank_average(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Mean of each score's percentile rank within its day's universe; NaN outside it."""
    inside = frame["in_universe"].astype(bool)
    ranks = [frame[name].where(inside).groupby(frame["date"]).rank(pct=True) for name in columns]
    return pd.concat(ranks, axis=1).mean(axis=1, skipna=False)


def evaluated_rows(frame: pd.DataFrame, folds: Sequence[Fold], horizon: int) -> pd.DataFrame:
    """Tradable universe rows of each fold's test_start..eval_end with a known rank label."""
    parts = []
    for fold in folds:
        inside = frame[(frame["fold"] == fold.index) & (frame["date"] <= fold.eval_end)]
        labels = load_labels(fold.test_start, fold.eval_end, horizon)
        parts.append(training_rows(inside, labels, RANK_LABEL))
    return pd.concat(parts, ignore_index=True)


def _scores(run_key: str, folds: Sequence[Fold], horizon: int) -> pd.DataFrame:
    """Both models' cached scores, refused unless built on the same partitions and rows."""
    split = [
        fold.index
        for fold in folds
        if fold_notes(run_key, BASELINE, fold.index).get("feature_hash")
        != fold_notes(run_key, MODEL, fold.index).get("feature_hash")
    ]
    if split:
        raise ValueError(
            f"folds {split} of {run_key} cache Ridge and LightGBM from different feature "
            "partitions; rerun train for both"
        )
    keys = [*ROW_KEYS, "fold"]
    ridge = load_predictions(run_key, BASELINE, folds)[[*keys, f"ridge_h{horizon}"]]
    boosted = load_predictions(run_key, MODEL, folds)[[*keys, f"{MODEL}_h{horizon}"]]
    frame = ridge.merge(boosted, on=keys, validate="one_to_one")
    if not len(frame) == len(ridge) == len(boosted):
        raise ValueError(f"the Ridge and LightGBM caches of {run_key} hold different rows")
    return frame


def ensemble_ics(run_key: str, folds: Sequence[Fold], horizon: int) -> dict[str, float]:
    """Daily OOS rank IC of Ridge, LightGBM and their rank average, pooled over the folds."""
    ridge, boosted = f"ridge_h{horizon}", f"{MODEL}_h{horizon}"
    frame = _scores(run_key, folds, horizon)
    frame = frame.assign(**{ENSEMBLE: rank_average(frame, [ridge, boosted])})
    rows = evaluated_rows(frame, folds, horizon)
    columns = {"ridge": ridge, MODEL: boosted, ENSEMBLE: ENSEMBLE}
    return {name: float(daily_rank_ic(rows, column).mean()) for name, column in columns.items()}


def adopt_ensemble(ics: Mapping[str, float]) -> bool:
    """The ensemble replaces LightGBM only when its research rank IC is strictly higher."""
    return bool(ics[ENSEMBLE] > ics[MODEL])


def ensemble_report(run_key: str, ics: Mapping[int, Mapping[str, float]]) -> str:
    """Markdown table of each horizon's pooled research rank IC and the ensemble verdict."""
    lines = [
        f"# Rank-average ensemble: {run_key}",
        "",
        "| horizon | Ridge IC | LightGBM IC | ensemble IC | verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for horizon, values in ics.items():
        verdict = "adopt" if adopt_ensemble(values) else "keep LightGBM alone"
        lines.append(
            f"| {horizon} | {values['ridge']:.4f} | {values[MODEL]:.4f} | "
            f"{values[ENSEMBLE]:.4f} | {verdict} |"
        )
    return "\n".join(lines) + "\n"
