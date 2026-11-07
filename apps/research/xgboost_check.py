"""XGBoost refits of the LightGBM rank folds, compared to catch implementation errors."""

import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import xgboost as xgb

from apps.research import lgbm_runs
from apps.research.gbdt import EARLY_STOPPING_ROUNDS, DailyRankIC, split_validation
from apps.research.gbdt import MAX_ROUNDS as DEFAULT_MAX_ROUNDS
from apps.research.labels import HORIZONS
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import HoldoutScope, ModelObjective, ModelRun
from apps.research.prediction_cache import (
    MissingFoldsError,
    fold_notes,
    fold_path,
    load_fold_predictions,
)
from apps.research.rank_model import RANK_LABEL
from apps.research.runs import feature_hash
from apps.research.training import KEYS, load_labels, training_rows
from apps.research.walkforward import Fold

GAP_LIMIT = 0.2
MAX_ROUNDS = DEFAULT_MAX_ROUNDS
XGB_PARAMS = {
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 200,
    "objective": "reg:squarederror",
}
DAILY_COLUMNS = ["fold", "horizon", "date", "lightgbm", "xgboost"]


def relative_gap(lightgbm: float, other: float) -> float:
    """|other − LightGBM| over |LightGBM|; NaN when LightGBM's IC is zero."""
    if lightgbm == 0:
        return math.nan
    return abs(other - lightgbm) / abs(lightgbm)


def fit_xgboost(rows: pd.DataFrame, fold: Fold, features: Sequence[str], seed: int) -> xgb.Booster:
    """XGBoost stopped on validation daily rank IC, then refit on every row for that many rounds."""
    fit, valid = split_validation(rows, fold)
    feval = DailyRankIC(valid["date"], valid[RANK_LABEL])
    params = {**XGB_PARAMS, "seed": seed, "disable_default_eval_metric": 1}
    names = list(features)
    probe = xgb.train(
        params,
        xgb.DMatrix(fit[names], label=fit[RANK_LABEL]),
        num_boost_round=MAX_ROUNDS,
        evals=[(xgb.DMatrix(valid[names], label=valid[RANK_LABEL]), "valid")],
        custom_metric=lambda scores, data: feval(scores, data)[:2],
        maximize=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    full = xgb.DMatrix(rows[names], label=rows[RANK_LABEL])
    return xgb.train(params, full, num_boost_round=probe.best_iteration + 1)


def _features(run_key: str, folds: Sequence[Fold]) -> dict[int, list[str]]:
    """Each fold's LightGBM rank features; a missing run is an error."""
    wanted = {
        lgbm_runs.run_id(run_key, ModelObjective.RANK, horizon, fold): fold.index
        for fold in folds
        for horizon in HORIZONS
    }
    found = {run.run_id: run for run in ModelRun.objects.filter(run_id__in=list(wanted))}
    missing = [run_id for run_id in wanted if run_id not in found]
    if missing:
        raise LookupError(f"no LightGBM rank run {missing}; run train --model lightgbm first")
    return {index: list(found[run_id].feature_list) for run_id, index in wanted.items()}


def _stale(run_key: str, fold: Fold, digest: str) -> bool:
    """Whether the fold's features or labels differ from those its LightGBM cache was fit on."""
    notes = fold_notes(run_key, lgbm_runs.MODEL, fold.index)
    current = {
        "feature_hash": digest,
        "label_hash": lgbm_runs.label_digest(HoldoutScope.RANK, fold),
    }
    return {key: notes.get(key) for key in current} != current


def _refuse(run_key: str, stale: list[int]) -> None:
    if stale:
        raise LookupError(
            f"folds {stale} of {run_key} are stale: their features or labels changed since "
            "LightGBM ran; rerun train --model lightgbm"
        )


def _check_caches(run_key: str, folds: Sequence[Fold], directory: Path) -> None:
    """Refuse missing LightGBM caches first, then folds whose inputs changed since."""
    absent = [
        fold.index for fold in folds if not fold_path(run_key, lgbm_runs.MODEL, fold.index).exists()
    ]
    if absent:
        raise MissingFoldsError(run_key, lgbm_runs.MODEL, absent)
    digests = {
        fold.index: feature_hash(fold.train_start, fold.test_end, directory) for fold in folds
    }
    _refuse(run_key, [fold.index for fold in folds if _stale(run_key, fold, digests[fold.index])])


def _window(inputs: pd.DataFrame, labels: pd.DataFrame, fold: Fold, test: bool) -> pd.DataFrame:
    start, end = (fold.test_start, fold.eval_end) if test else (fold.train_start, fold.train_end)
    return training_rows(inputs[inputs["date"].between(start, end)], labels, RANK_LABEL)


def _fold_ics(
    run_key: str, fold: Fold, features: list[str], directory: Path, seed: int
) -> list[pd.DataFrame]:
    read = lgbm_runs.rank_fold_inputs(fold, directory, features)
    _refuse(run_key, [fold.index] if _stale(run_key, fold, read.feature_hash) else [])
    inputs = read.raw
    cached = load_fold_predictions(run_key, lgbm_runs.MODEL, fold)
    parts = []
    for horizon in HORIZONS:
        column = f"{lgbm_runs.MODEL}_h{horizon}"
        labels = load_labels(fold.train_start, fold.eval_end, horizon)
        booster = fit_xgboost(_window(inputs, labels, fold, test=False), fold, features, seed)
        evaluated = _window(inputs, labels, fold, test=True).merge(
            cached[[*KEYS, column]], on=KEYS, validate="one_to_one"
        )
        scored = evaluated.assign(xgboost=booster.predict(xgb.DMatrix(evaluated[features])))
        both = pd.concat(
            [daily_rank_ic(scored, column).rename("lightgbm"), daily_rank_ic(scored, "xgboost")],
            axis=1,
        ).rename(columns={"ic": "xgboost"})
        parts.append(both.reset_index().assign(fold=fold.index, horizon=horizon))
    _refuse(run_key, [fold.index] if _stale(run_key, fold, read.feature_hash) else [])
    return parts


def xgboost_daily_ics(
    run_key: str, folds: Sequence[Fold], directory: Path, seed: int
) -> pd.DataFrame:
    """Daily OOS rank IC of LightGBM's cached scores and an XGBoost refit, on the same rows."""
    features = _features(run_key, folds)
    _check_caches(run_key, folds, directory)
    parts = [
        part
        for fold in folds
        for part in _fold_ics(run_key, fold, features[fold.index], directory, seed)
    ]
    return pd.concat(parts, ignore_index=True)[DAILY_COLUMNS]


def xgboost_report(run_key: str, daily: pd.DataFrame) -> str:
    """Markdown summary: pooled daily IC per horizon with its gap, then each fold's mean."""
    lines = [
        f"# XGBoost cross-check: {run_key}",
        "",
        "| horizon | LightGBM IC | XGBoost IC | gap | check |",
        "| --- | --- | --- | --- | --- |",
    ]
    pooled = daily.groupby("horizon", sort=True)[["lightgbm", "xgboost"]].mean()
    for horizon, row in pooled.iterrows():
        gap = relative_gap(float(row["lightgbm"]), float(row["xgboost"]))
        shown = "n/a" if math.isnan(gap) else f"{gap:.1%}"
        check = "ok" if gap <= GAP_LIMIT else "investigate the implementation"
        lines.append(
            f"| {horizon} | {row['lightgbm']:.4f} | {row['xgboost']:.4f} | {shown} | {check} |"
        )
    lines += ["", "## Per fold", "", "| fold | horizon | LightGBM IC | XGBoost IC |"]
    lines.append("| --- | --- | --- | --- |")
    by_fold = daily.groupby(["fold", "horizon"], sort=True)[["lightgbm", "xgboost"]].mean()
    for fold, horizon, lightgbm, other in by_fold.reset_index().itertuples(index=False):
        lines.append(f"| {fold} | {horizon} | {lightgbm:.4f} | {other:.4f} |")
    return "\n".join(lines) + "\n"
