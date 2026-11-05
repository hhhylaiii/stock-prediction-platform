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
SCORES = ["ridge", MODEL, ENSEMBLE]
DAILY_COLUMNS = ["fold", "horizon", "date", *SCORES]


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


def evaluated_rows(frame: pd.DataFrame, fold: Fold, horizon: int) -> pd.DataFrame:
    """Tradable universe rows of the fold's test_start..eval_end with a known rank label."""
    inside = frame[(frame["fold"] == fold.index) & (frame["date"] <= fold.eval_end)]
    labels = load_labels(fold.test_start, fold.eval_end, horizon)
    return training_rows(inside, labels, RANK_LABEL)


def _fingerprint(run_key: str, model: str, fold: Fold) -> dict[str, str | None]:
    notes = fold_notes(run_key, model, fold.index)
    return {key: notes.get(key) for key in ("feature_hash", "label_hash")}


def _scores(run_key: str, folds: Sequence[Fold], horizon: int) -> pd.DataFrame:
    """Both models' cached scores, refused unless fit on the same features, labels and rows."""
    keys = [*ROW_KEYS, "fold"]
    ridge = load_predictions(run_key, BASELINE, folds)[[*keys, f"ridge_h{horizon}"]]
    boosted = load_predictions(run_key, MODEL, folds)[[*keys, f"{MODEL}_h{horizon}"]]
    split = [
        fold.index
        for fold in folds
        if _fingerprint(run_key, BASELINE, fold) != _fingerprint(run_key, MODEL, fold)
    ]
    if split:
        raise ValueError(
            f"folds {split} of {run_key} cache Ridge and LightGBM fit on different features "
            "or labels; rerun train for both"
        )
    frame = ridge.merge(boosted, on=keys, validate="one_to_one")
    if not len(frame) == len(ridge) == len(boosted):
        raise ValueError(f"the Ridge and LightGBM caches of {run_key} hold different rows")
    return frame


def ensemble_daily_ics(run_key: str, folds: Sequence[Fold], horizon: int) -> pd.DataFrame:
    """Each fold's daily OOS rank IC of Ridge, LightGBM and their rank average."""
    ridge, boosted = f"ridge_h{horizon}", f"{MODEL}_h{horizon}"
    frame = _scores(run_key, folds, horizon)
    frame = frame.assign(**{ENSEMBLE: rank_average(frame, [ridge, boosted])})
    columns = dict(zip(SCORES, [ridge, boosted, ENSEMBLE], strict=True))
    parts = []
    for fold in folds:
        rows = evaluated_rows(frame, fold, horizon)
        ics = [daily_rank_ic(rows, column).rename(name) for name, column in columns.items()]
        parts.append(pd.concat(ics, axis=1).reset_index().assign(fold=fold.index, horizon=horizon))
    return pd.concat(parts, ignore_index=True)[DAILY_COLUMNS]


def adopt_ensemble(ics: Mapping[str, float]) -> bool:
    """The ensemble replaces LightGBM only when its research rank IC is strictly higher."""
    return bool(ics[ENSEMBLE] > ics[MODEL])


def ensemble_report(run_key: str, daily: pd.DataFrame) -> str:
    """Markdown summary: pooled daily IC per horizon with the verdict, then each fold's mean."""
    lines = [
        f"# Rank-average ensemble: {run_key}",
        "",
        "| horizon | Ridge IC | LightGBM IC | ensemble IC | verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    pooled = daily.groupby("horizon", sort=True)[SCORES].mean().reset_index()
    for horizon, ridge, lightgbm, ensemble in pooled.itertuples(index=False):
        adopted = adopt_ensemble({MODEL: lightgbm, ENSEMBLE: ensemble})
        verdict = "adopt" if adopted else "keep LightGBM alone"
        lines.append(f"| {horizon} | {ridge:.4f} | {lightgbm:.4f} | {ensemble:.4f} | {verdict} |")
    lines += ["", "## Per fold", "", "| fold | horizon | Ridge IC | LightGBM IC | ensemble IC |"]
    lines.append("| --- | --- | --- | --- | --- |")
    by_fold = daily.groupby(["fold", "horizon"], sort=True)[SCORES].mean().reset_index()
    for fold, horizon, ridge, lightgbm, ensemble in by_fold.itertuples(index=False):
        lines.append(f"| {fold} | {horizon} | {ridge:.4f} | {lightgbm:.4f} | {ensemble:.4f} |")
    return "\n".join(lines) + "\n"
