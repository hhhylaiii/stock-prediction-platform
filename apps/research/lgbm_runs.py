"""Walk-forward LightGBM rank folds: boosters, model_run rows and cached predictions (SPEC §7)."""

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from apps.research.baseline_runs import (
    ROW_KEYS,
    check_run_key,
    fold_inputs,
    label_digest,
    require_candidates,
)
from apps.research.gbdt import (
    BASE_PARAMS,
    EARLY_STOPPING_ROUNDS,
    FEVAL_EVERY,
    Boosted,
    seeded,
)
from apps.research.gbdt import MAX_ROUNDS as DEFAULT_MAX_ROUNDS
from apps.research.holdout import HoldoutError, is_holdout
from apps.research.labels import HORIZONS
from apps.research.metrics.ic import daily_rank_ic, summarize_ic
from apps.research.models import HoldoutScope, ModelObjective, ModelRun
from apps.research.prediction_cache import (
    fold_notes,
    fold_path,
    missing_folds,
    save_fold_predictions,
)
from apps.research.rank_model import MODEL_TYPE, RANK_LABEL, fit_rank, rank_features, rank_inputs
from apps.research.runs import feature_hash, lib_versions, record_run, screened_features
from apps.research.training import load_labels, training_rows
from apps.research.walkforward import Fold

MODEL = MODEL_TYPE
RECIPE_VERSION = "1"
MAX_ROUNDS = DEFAULT_MAX_ROUNDS
SCOPE = HoldoutScope.RANK


def recipe(features: list[str], seed: int) -> str:
    """RECIPE_VERSION with a digest of the features, LightGBM settings and seed of a fit."""
    settings = {
        "features": features,
        "params": BASE_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "feval_every": FEVAL_EVERY,
        "max_rounds": MAX_ROUNDS,
        "seed": seed,
    }
    listed = json.dumps(settings, sort_keys=True).encode()
    return f"{RECIPE_VERSION}-{hashlib.sha256(listed).hexdigest()[:12]}"


def _current_notes(fold: Fold, directory: Path, features: list[str], seed: int) -> dict[str, str]:
    """Notes a fold cached now would carry: its features, labels and recipe."""
    return {
        "feature_hash": feature_hash(fold.train_start, fold.test_end, directory),
        "label_hash": label_digest(SCOPE, fold),
        "recipe": recipe(features, seed),
    }


def _plain(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _oos(scores: pd.Series, evaluated: pd.DataFrame, horizon: int) -> dict[str, float | None]:
    """The fold's OOS daily rank IC mean, ICIR and Newey-West t; empty without rows."""
    if evaluated.empty:
        return {"ic_mean": None, "icir": None, "ic_nw_t": None}
    summary = summarize_ic(daily_rank_ic(evaluated.assign(_score=scores), "_score"), lag=horizon)
    return {
        "ic_mean": _plain(summary.mean),
        "icir": _plain(summary.icir),
        "ic_nw_t": _plain(summary.nw_t),
    }


def _run(
    fold: Fold,
    run_key: str,
    horizon: int,
    features: list[str],
    seed: int,
    boosted: Boosted,
    oos: dict[str, float | None],
    digest: str,
) -> ModelRun:
    gains = boosted.booster.feature_importance("gain")
    params = {
        "horizon": horizon,
        "label": RANK_LABEL,
        "rounds": boosted.rounds,
        "valid_ic": _plain(boosted.valid_score),
        "lightgbm": {**BASE_PARAMS, **seeded(seed)},
        "max_rounds": MAX_ROUNDS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "feval_every": FEVAL_EVERY,
        "ic_nw_t": oos["ic_nw_t"],
        "importance_gain": {name: float(gain) for name, gain in zip(features, gains, strict=True)},
    }
    return ModelRun(
        run_id=f"{run_key}.{MODEL}.h{horizon}.f{fold.index:03d}",
        run_key=run_key,
        model_type=MODEL,
        objective=ModelObjective.RANK,
        version=RECIPE_VERSION,
        train_start=fold.train_start,
        train_end=fold.train_end,
        params=params,
        feature_list=features,
        seed=seed,
        lib_versions=lib_versions(),
        feature_hash=digest,
        ic_mean=oos["ic_mean"],
        icir=oos["icir"],
    )


def run_fold(fold: Fold, run_key: str, directory: Path, features: list[str], seed: int) -> Path:
    """Fit one fold's 5- and 20-day rank boosters; the old cache goes first, the new one last."""
    if is_holdout(fold):
        raise HoldoutError(f"fold {fold.index} is holdout; it runs once with the frozen design")
    require_candidates(run_key, fold)
    fold_path(run_key, MODEL, fold.index).unlink(missing_ok=True)
    labels_digest = label_digest(SCOPE, fold)
    read = fold_inputs(fold, fold.train_start, fold.test_end, directory, features)
    inputs = rank_inputs(read.raw, features)
    test = inputs[read.test]
    evaluated_days = inputs[read.window(fold.test_start, fold.eval_end)]
    columns, trained = {}, []
    for horizon in HORIZONS:
        labels = load_labels(fold.train_start, fold.eval_end, horizon)
        boosted = fit_rank(inputs, labels, fold, features, seed, max_rounds=MAX_ROUNDS)
        columns[f"{MODEL}_h{horizon}"] = boosted.booster.predict(test[features])
        evaluated = training_rows(evaluated_days, labels, RANK_LABEL)
        scores = pd.Series(boosted.booster.predict(evaluated[features]), index=evaluated.index)
        run = _run(
            fold,
            run_key,
            horizon,
            features,
            seed,
            boosted,
            _oos(scores, evaluated, horizon),
            read.feature_hash,
        )
        trained.append((run, boosted.booster))
    if label_digest(SCOPE, fold) != labels_digest:
        raise RuntimeError(f"label_daily changed while fitting fold {fold.index}; run it again")
    for run, booster in trained:
        record_run(run, booster, fold=fold)
    rows = read.raw.loc[read.test, ROW_KEYS].reset_index(drop=True)
    notes = {
        "feature_hash": read.feature_hash,
        "label_hash": labels_digest,
        "recipe": recipe(features, seed),
    }
    return save_fold_predictions(run_key, MODEL, fold, rows.assign(**columns), notes=notes)


def _stale(run_key: str, fold: Fold, directory: Path, features: list[str], seed: int) -> bool:
    """Whether a cached fold's features, labels or recipe have changed since it was fitted."""
    return fold_notes(run_key, MODEL, fold.index) != _current_notes(fold, directory, features, seed)


def run_folds(
    folds: Iterable[Fold], run_key: str, directory: Path, seed: int = 0, rerun: bool = False
) -> list[int]:
    """Run rank folds without a current cache, or all with rerun; holdout folds are refused."""
    check_run_key(SCOPE, run_key)
    chosen = list(folds)
    held = [fold.index for fold in chosen if is_holdout(fold)]
    if held:
        raise HoldoutError(f"folds {held} are holdout; they run once with the frozen design")
    features = rank_features(screened_features())
    missing = set(missing_folds(run_key, MODEL, chosen))
    ran = []
    for fold in chosen:
        if rerun or fold.index in missing or _stale(run_key, fold, directory, features, seed):
            run_fold(fold, run_key, directory, features, seed)
            ran.append(fold.index)
    return ran
