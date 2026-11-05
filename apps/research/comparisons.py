"""Cross-checks of the LightGBM rank folds: a rank-average ensemble and XGBoost refits."""

import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import xgboost as xgb

from apps.market.calendar import TradingDays
from apps.research import lgbm_runs
from apps.research.baseline_runs import BASELINE, ROW_KEYS
from apps.research.features.store import read_partitions
from apps.research.gbdt import EARLY_STOPPING_ROUNDS, DailyRankIC, split_validation
from apps.research.gbdt import MAX_ROUNDS as DEFAULT_MAX_ROUNDS
from apps.research.labels import HORIZONS, latest_price_date
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import ModelObjective, ModelRun
from apps.research.prediction_cache import load_predictions
from apps.research.rank_model import RANK_LABEL, rank_inputs
from apps.research.training import load_labels, training_rows
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

GAP_LIMIT = 0.2
ENSEMBLE = "ensemble"
MODEL = lgbm_runs.MODEL
MAX_ROUNDS = DEFAULT_MAX_ROUNDS
XGB_PARAMS = {
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "objective": "reg:squarederror",
}
FOLD_COLUMNS = ["fold", "horizon", "lightgbm_ic", "xgboost_ic"]


def research_folds(config: WalkForwardConfig, last: date | None) -> list[Fold]:
    """Research folds up to last, or up to the latest price day; tuning and holdout stay out."""
    end = last or latest_price_date()
    if end is None:
        raise ValueError("daily_price is empty")
    return [fold for fold in generate_folds(TradingDays.from_db(), end, config) if fold.research]


def rank_average(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Mean of each score's percentile rank within its day's universe; NaN outside it."""
    inside = frame["in_universe"].astype(bool)
    ranks = [frame[name].where(inside).groupby(frame["date"]).rank(pct=True) for name in columns]
    return pd.concat(ranks, axis=1).mean(axis=1, skipna=False)


def relative_gap(lightgbm: float, other: float) -> float:
    """|other − LightGBM| over |LightGBM|; NaN when LightGBM's IC is zero."""
    if lightgbm == 0:
        return math.nan
    return abs(other - lightgbm) / abs(lightgbm)


def adopt_ensemble(ics: Mapping[str, float]) -> bool:
    """The ensemble replaces LightGBM only when its research rank IC is strictly higher."""
    return bool(ics[ENSEMBLE] > ics[MODEL])


def _evaluated(frame: pd.DataFrame, folds: Sequence[Fold], horizon: int) -> pd.DataFrame:
    """Tradable universe rows of each fold's test_start..eval_end with a known rank label."""
    parts = []
    for fold in folds:
        inside = frame[(frame["fold"] == fold.index) & (frame["date"] <= fold.eval_end)]
        labels = load_labels(fold.test_start, fold.eval_end, horizon)
        parts.append(training_rows(inside, labels, RANK_LABEL))
    return pd.concat(parts, ignore_index=True)


def ensemble_ics(run_key: str, folds: Sequence[Fold], horizon: int) -> dict[str, float]:
    """Mean daily OOS rank IC of Ridge, LightGBM and their rank average over the folds."""
    ridge, boosted = f"ridge_h{horizon}", f"{MODEL}_h{horizon}"
    keys = [*ROW_KEYS, "fold"]
    base = load_predictions(run_key, BASELINE, folds)[[*keys, ridge]]
    lgbm = load_predictions(run_key, MODEL, folds)[[*keys, boosted]]
    frame = base.merge(lgbm, on=keys, validate="one_to_one")
    frame = frame.assign(**{ENSEMBLE: rank_average(frame, [ridge, boosted])})
    rows = _evaluated(frame, folds, horizon)
    columns = {"ridge": ridge, MODEL: boosted, ENSEMBLE: ENSEMBLE}
    return {name: float(daily_rank_ic(rows, column).mean()) for name, column in columns.items()}


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
    rounds = probe.best_iteration + 1
    full = xgb.DMatrix(rows[names], label=rows[RANK_LABEL])
    return xgb.train(params, full, num_boost_round=rounds)


def _lightgbm_runs(run_key: str, folds: Sequence[Fold]) -> dict[str, ModelRun]:
    """The LightGBM rank run of every fold and horizon; a missing one is an error."""
    wanted = [
        lgbm_runs.run_id(run_key, ModelObjective.RANK, horizon, fold)
        for fold in folds
        for horizon in HORIZONS
    ]
    found = {run.run_id: run for run in ModelRun.objects.filter(run_id__in=wanted)}
    missing = [run_id for run_id in wanted if run_id not in found]
    if missing:
        raise LookupError(f"no LightGBM rank run {missing}; run train --model lightgbm first")
    return found


def _window(inputs: pd.DataFrame, labels: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return training_rows(inputs[inputs["date"].between(start, end)], labels, RANK_LABEL)


def xgboost_fold_ics(
    run_key: str, folds: Sequence[Fold], directory: Path, seed: int
) -> pd.DataFrame:
    """Per fold and horizon, LightGBM's recorded OOS IC beside an XGBoost refit's on its inputs."""
    runs = _lightgbm_runs(run_key, folds)
    records = []
    for fold in folds:
        first = runs[lgbm_runs.run_id(run_key, ModelObjective.RANK, HORIZONS[0], fold)]
        features = list(first.feature_list)
        inputs = rank_inputs(read_partitions(fold.train_start, fold.test_end, directory), features)
        for horizon in HORIZONS:
            run = runs[lgbm_runs.run_id(run_key, ModelObjective.RANK, horizon, fold)]
            labels = load_labels(fold.train_start, fold.eval_end, horizon)
            booster = fit_xgboost(
                _window(inputs, labels, fold.train_start, fold.train_end), fold, features, seed
            )
            evaluated = _window(inputs, labels, fold.test_start, fold.eval_end)
            scores = booster.predict(xgb.DMatrix(evaluated[features]))
            ic = daily_rank_ic(evaluated.assign(_xgb=scores), "_xgb").mean()
            lightgbm = math.nan if run.ic_mean is None else run.ic_mean
            records.append([fold.index, horizon, lightgbm, float(ic)])
    return pd.DataFrame(records, columns=FOLD_COLUMNS)


def ensemble_report(run_key: str, ics: Mapping[int, Mapping[str, float]]) -> str:
    """Markdown table of each horizon's research rank IC and the ensemble verdict."""
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


def xgboost_report(run_key: str, table: pd.DataFrame) -> str:
    """Markdown summary of the XGBoost cross-check: mean ICs, their gap and every fold."""
    lines = [
        f"# XGBoost cross-check: {run_key}",
        "",
        "| horizon | LightGBM IC | XGBoost IC | gap | check |",
        "| --- | --- | --- | --- | --- |",
    ]
    for horizon, group in table.groupby("horizon", sort=True):
        lightgbm, other = float(group["lightgbm_ic"].mean()), float(group["xgboost_ic"].mean())
        gap = relative_gap(lightgbm, other)
        check = "ok" if gap <= GAP_LIMIT else "investigate the implementation"
        lines.append(f"| {horizon} | {lightgbm:.4f} | {other:.4f} | {gap:.1%} | {check} |")
    lines += ["", "## Per fold", "", "| fold | horizon | LightGBM IC | XGBoost IC |"]
    lines.append("| --- | --- | --- | --- |")
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.horizon} | {row.lightgbm_ic:.4f} | {row.xgboost_ic:.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(text: str, output: str | None) -> str | None:
    """Write a report to output when given; the path written, or None to print instead."""
    if output is None:
        return None
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)
