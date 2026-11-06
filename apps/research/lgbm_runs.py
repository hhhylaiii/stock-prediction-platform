"""Walk-forward LightGBM folds of each scope: boosters, model_run rows and cached predictions."""

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from apps.research import target_models
from apps.research.baseline_runs import (
    RETURN_HORIZON,
    ROW_KEYS,
    FoldInputs,
    check_run_key,
    fold_inputs,
    require_candidates,
)
from apps.research.features.store import read_partitions
from apps.research.features.transforms import MarketBins
from apps.research.gbdt import (
    BASE_PARAMS,
    EARLY_STOPPING_ROUNDS,
    FEVAL_EVERY,
    Bagged,
    Boosted,
    bag_seeds,
    seeded,
)
from apps.research.gbdt import MAX_ROUNDS as DEFAULT_MAX_ROUNDS
from apps.research.holdout import HoldoutError, is_holdout
from apps.research.labels import HORIZONS, VOL_DAYS
from apps.research.metrics.forecast import (
    expected_calibration_error,
    pinball_loss,
    reliability_table,
)
from apps.research.metrics.ic import daily_rank_ic, summarize_ic
from apps.research.models import HoldoutScope, ModelObjective, ModelRun
from apps.research.prediction_cache import (
    fold_notes,
    fold_path,
    missing_folds,
    save_fold_predictions,
)
from apps.research.ranges import years
from apps.research.rank_model import (
    GRADE,
    LAMBDARANK,
    LEVELS,
    MODEL_TYPE,
    RANK_LABEL,
    SPEC_RECIPE,
    Variant,
    fit_rank,
    rank_features,
    rank_inputs,
)
from apps.research.runs import (
    RunError,
    feature_hash,
    label_hash,
    lib_versions,
    record_run,
    screened_features,
)
from apps.research.target_models import (
    MARKET_LEVEL,
    RETURN_LABEL,
    TARGET_RECIPE,
    UP_LABEL,
    VOL_LABEL,
    TargetVariant,
    fit_probability,
    fit_quantiles,
    fit_return,
    fit_volatility,
    raw_features,
    raw_inputs,
    sorted_quantiles,
    volatility,
)
from apps.research.training import load_labels, training_rows
from apps.research.walkforward import Fold

MODEL = MODEL_TYPE
RANK_FIELDS = (RANK_LABEL, "untradable_t1")
RECIPE_VERSION = "4"
LATER_DEFAULTS: dict[str, Any] = {"min_rounds": 0, "fixed_rounds": ()}
MAX_ROUNDS = DEFAULT_MAX_ROUNDS
SCOPE_HORIZONS: dict[str, tuple[int, ...]] = {
    HoldoutScope.RANK: HORIZONS,
    HoldoutScope.RETURN: (RETURN_HORIZON,),
    HoldoutScope.INTERVAL: (VOL_DAYS,),
}
INTERVAL_TAGS = {
    ModelObjective.QUANTILE: "q",
    ModelObjective.VOLATILITY: "vol",
    ModelObjective.CLASSIFICATION: "prob",
}
NO_IC: Mapping[str, float | None] = {"ic_mean": None, "icir": None, "ic_nw_t": None}


@dataclass(frozen=True)
class Job:
    fold: Fold
    run_key: str
    read: FoldInputs
    features: list[str]
    seed: int
    bags: int
    variant: Variant = SPEC_RECIPE


@dataclass(frozen=True)
class FoldFit:
    columns: dict[str, np.ndarray]
    trained: list[tuple[ModelRun, Any]]


def run_id(run_key: str, objective: str, horizon: int, fold: Fold) -> str:
    """run_id of one fold's booster; interval models share a horizon, so they carry a tag."""
    tag = INTERVAL_TAGS.get(ModelObjective(objective))
    middle = MODEL if tag is None else f"{MODEL}.{tag}"
    return f"{run_key}.{middle}.h{horizon}.f{fold.index:03d}"


def _targets(scope: str) -> dict[str, object]:
    """Target settings a scope's recipe digests besides the shared LightGBM settings."""
    if scope == HoldoutScope.RANK:
        return {"label": RANK_LABEL, "feval_every": FEVAL_EVERY}
    if scope == HoldoutScope.RETURN:
        return {"label": RETURN_LABEL, "huber_mad_scale": target_models.HUBER_MAD_SCALE}
    return {
        "quantiles": target_models.QUANTILES,
        "vol_floor": target_models.VOL_FLOOR,
        "ece_limit": target_models.ECE_LIMIT,
        "ece_bins": target_models.ECE_BINS,
    }


def variant_note(variant: Variant) -> dict[str, Any]:
    """The variant as recipes and params record it; later fields are left out at their defaults."""
    return {
        name: value
        for name, value in asdict(variant).items()
        if not (name in LATER_DEFAULTS and value == LATER_DEFAULTS[name])
    }


def recipe(
    scope: str,
    features: list[str],
    seed: int,
    bags: int = 1,
    variant: Variant = SPEC_RECIPE,
    target: TargetVariant = TARGET_RECIPE,
) -> str:
    """RECIPE_VERSION with a digest of a scope's features, targets, LightGBM settings and seeds."""
    settings: dict[str, Any] = {
        "scope": scope,
        "targets": _targets(scope),
        "features": features,
        "params": BASE_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "max_rounds": MAX_ROUNDS,
        "seed": seed,
        "bag_seeds": bag_seeds(seed, bags),
        "variant": variant_note(variant),
    }
    if target != TARGET_RECIPE:
        settings["target_variant"] = asdict(target)
    listed = json.dumps(settings, sort_keys=True).encode()
    return f"{RECIPE_VERSION}-{hashlib.sha256(listed).hexdigest()[:12]}"


def label_digest(scope: str, fold: Fold) -> str:
    """label_hash of the horizons a scope fits and evaluates on, train_start to eval_end."""
    return label_hash(fold.train_start, fold.eval_end, SCOPE_HORIZONS[scope])


def scope_features(scope: str, target: TargetVariant = TARGET_RECIPE) -> list[str]:
    """Features a scope's boosters read: stock-level, plus market ones unless rank or left out."""
    try:
        screened = screened_features()
    except RunError as error:
        raise RunError(f"the {scope} scope fits on the screened features: {error}") from error
    if scope == HoldoutScope.RANK:
        return rank_features(screened)
    features = raw_features(screened)
    return features if target.market else [name for name in features if name not in MARKET_LEVEL]


def input_notes(scope: str, fold: Fold, directory: Path) -> dict[str, str]:
    """Fingerprints of the feature partitions and labels a scope's fold reads today."""
    return {
        "feature_hash": feature_hash(fold.train_start, fold.test_end, directory),
        "label_hash": label_digest(scope, fold),
    }


def _current_notes(
    scope: str,
    fold: Fold,
    directory: Path,
    features: list[str],
    seed: int,
    bags: int,
    variant: Variant,
    target: TargetVariant,
) -> dict[str, str]:
    """Notes a fold cached now would carry: its features, labels and recipe."""
    return {
        **input_notes(scope, fold, directory),
        "recipe": recipe(scope, features, seed, bags, variant, target),
    }


def _plain(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _mean(values: np.ndarray) -> float | None:
    return _plain(float(np.mean(values)))


def _gains(booster: lgb.Booster | Bagged, features: list[str]) -> dict[str, float]:
    gains = booster.feature_importance("gain")
    return {name: float(gain) for name, gain in zip(features, gains, strict=True)}


def _edges(bins: MarketBins) -> dict[str, list[float | None]]:
    """Bin edges for params; a feature missing through the window has open edges."""
    return {name: [_plain(edge) for edge in edges] for name, edges in bins.edges.items()}


def _oos(scores: np.ndarray, evaluated: pd.DataFrame, horizon: int) -> Mapping[str, float | None]:
    """The fold's OOS daily rank IC mean, ICIR and Newey-West t."""
    summary = summarize_ic(daily_rank_ic(evaluated.assign(_score=scores), "_score"), lag=horizon)
    return {
        "ic_mean": _plain(summary.mean),
        "icir": _plain(summary.icir),
        "ic_nw_t": _plain(summary.nw_t),
    }


def _evaluated(job: Job, inputs: pd.DataFrame, labels: pd.DataFrame, label: str) -> pd.DataFrame:
    """Tradable universe rows of test_start..eval_end with a known label; none is an error."""
    inside = inputs[job.read.window(job.fold.test_start, job.fold.eval_end)]
    rows = training_rows(inside, labels, label)
    if rows.empty:
        raise RuntimeError(
            f"fold {job.fold.index} has no labelled {label} rows in test_start..eval_end; "
            "build labels first"
        )
    return rows


def _run(
    job: Job,
    objective: str,
    horizon: int,
    label: str,
    params: Mapping[str, object],
    oos: Mapping[str, float | None] = NO_IC,
) -> ModelRun:
    fold = job.fold
    shared = {
        "horizon": horizon,
        "label": label,
        "lightgbm": {**BASE_PARAMS, **seeded(job.seed)},
        "bags": job.bags,
        "bag_seeds": bag_seeds(job.seed, job.bags),
        "max_rounds": MAX_ROUNDS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }
    return ModelRun(
        run_id=run_id(job.run_key, objective, horizon, fold),
        run_key=job.run_key,
        model_type=MODEL,
        objective=objective,
        version=RECIPE_VERSION,
        train_start=fold.train_start,
        train_end=fold.train_end,
        params={**shared, **params},
        feature_list=job.features,
        seed=job.seed,
        lib_versions=lib_versions(),
        feature_hash=job.read.feature_hash,
        ic_mean=oos["ic_mean"],
        icir=oos["icir"],
    )


def rank_fold_inputs(fold: Fold, directory: Path, features: list[str]) -> FoldInputs:
    """Rank inputs of train_start..test_end as float32, z-scored a yearly partition at a time.

    raw holds the inputs themselves; a day's z-score reads that day only, so years split cleanly.
    """
    start, end = fold.train_start, fold.test_end
    digest = feature_hash(start, end, directory)
    types = dict.fromkeys(features, "float32")
    parts = [
        rank_inputs(read_partitions(low, high, directory), features)[[*ROW_KEYS, *features]].astype(
            types
        )
        for low, high in years(start, end)
    ]
    if feature_hash(start, end, directory) != digest:
        raise RuntimeError("feature partitions changed while reading; rerun after build_features")
    return FoldInputs(fold, pd.concat(parts, ignore_index=True), digest, features)


def _rank(job: Job) -> FoldFit:
    fold, read, features = job.fold, job.read, job.features
    inputs = read.raw
    columns, trained = {}, []
    for horizon in HORIZONS:
        labels = load_labels(fold.train_start, fold.eval_end, horizon, RANK_FIELDS)
        fixed = job.variant.fixed_for(horizon)
        boosted = fit_rank(
            inputs, labels, fold, features, job.seed, MAX_ROUNDS, job.bags, job.variant, fixed
        )
        test = inputs[read.test][features]
        columns[f"{MODEL}_h{horizon}"] = np.asarray(boosted.booster.predict(test))
        evaluated = _evaluated(job, inputs, labels, RANK_LABEL)
        scores = np.asarray(boosted.booster.predict(evaluated[features]))
        oos = _oos(scores, evaluated, horizon)
        ranking = job.variant.objective == "lambdarank"
        params = {
            "rounds": boosted.rounds,
            "best_round": boosted.best,
            "valid_ic": _plain(boosted.valid_score),
            "feval_every": FEVAL_EVERY,
            "variant": variant_note(job.variant),
            "lightgbm": {
                **BASE_PARAMS,
                "objective": job.variant.objective,
                **(LAMBDARANK if ranking else {}),
                **seeded(job.seed),
            },
            **({"graded_from": RANK_LABEL, "levels": LEVELS} if ranking else {}),
            "ic_nw_t": oos["ic_nw_t"],
            "importance_gain": _gains(boosted.booster, features),
        }
        label = GRADE if ranking else RANK_LABEL
        run = _run(job, ModelObjective.RANK, horizon, label, params, oos)
        trained.append((run, boosted.booster))
    return FoldFit(columns, trained)


def _return(job: Job) -> FoldFit:
    fold, features = job.fold, job.features
    inputs, bins = raw_inputs(job.read.raw, fold, features)
    labels = load_labels(fold.train_start, fold.eval_end, RETURN_HORIZON)
    fitted = fit_return(inputs, labels, fold, features, job.seed, MAX_ROUNDS, job.bags)
    booster = fitted.boosted.booster
    evaluated = _evaluated(job, inputs, labels, RETURN_LABEL)
    predicted = np.asarray(booster.predict(evaluated[features]))
    actual = evaluated[RETURN_LABEL].to_numpy(np.float64)
    oos = _oos(predicted, evaluated, RETURN_HORIZON)
    params = {
        "rounds": fitted.boosted.rounds,
        "valid_huber": _plain(fitted.boosted.valid_score),
        "delta": fitted.delta,
        "market_bins": _edges(bins),
        "oos_mae": _mean(np.abs(predicted - actual)),
        "oos_mae_zero": _mean(np.abs(actual)),
        "ic_nw_t": oos["ic_nw_t"],
        "importance_gain": _gains(booster, features),
    }
    run = _run(job, ModelObjective.REGRESSION, RETURN_HORIZON, RETURN_LABEL, params, oos)
    model = {"booster": booster, "market_bins": dict(bins.edges)}
    column = np.asarray(booster.predict(inputs[job.read.test][features]))
    return FoldFit({f"{MODEL}_ret_h{RETURN_HORIZON}": column}, [(run, model)])


def _quantiles(boosters: Mapping[str, Boosted], frame: pd.DataFrame, features: list[str]) -> Any:
    raw = {name: np.asarray(fit.booster.predict(frame[features])) for name, fit in boosters.items()}
    return sorted_quantiles(raw)


def _quantile(job: Job, inputs: pd.DataFrame, labels: pd.DataFrame, bins: MarketBins) -> FoldFit:
    fold, features = job.fold, job.features
    boosters = fit_quantiles(inputs, labels, fold, features, job.seed, MAX_ROUNDS, job.bags)
    evaluated = _evaluated(job, inputs, labels, RETURN_LABEL)
    scored = _quantiles(boosters, evaluated, features)
    actual = evaluated[RETURN_LABEL].to_numpy(np.float64)
    pinball = [
        pinball_loss(actual, scored[name], alpha).mean()
        for name, alpha in target_models.QUANTILES.items()
    ]
    params = {
        "rounds": {name: fit.rounds for name, fit in boosters.items()},
        "valid_loss": {name: _plain(fit.valid_score) for name, fit in boosters.items()},
        "market_bins": _edges(bins),
        "oos_coverage": _mean((actual >= scored["q10"]) & (actual <= scored["q90"])),
        "oos_pinball": _plain(float(sum(pinball))),
        "importance_gain": _gains(boosters["q50"].booster, features),
    }
    run = _run(job, ModelObjective.QUANTILE, VOL_DAYS, RETURN_LABEL, params)
    model = {
        "boosters": {name: fit.booster for name, fit in boosters.items()},
        "market_bins": dict(bins.edges),
    }
    test = _quantiles(boosters, inputs[job.read.test], features)
    columns = {f"{MODEL}_{name}_h{VOL_DAYS}": values for name, values in test.items()}
    return FoldFit(columns, [(run, model)])


def _volatility(job: Job, inputs: pd.DataFrame, labels: pd.DataFrame, bins: MarketBins) -> FoldFit:
    fold, features = job.fold, job.features
    boosted = fit_volatility(inputs, labels, fold, features, job.seed, MAX_ROUNDS, job.bags)
    evaluated = _evaluated(job, inputs, labels, VOL_LABEL)
    error = volatility(boosted, evaluated, features) - evaluated[VOL_LABEL].to_numpy(np.float64)
    params = {
        "rounds": boosted.rounds,
        "valid_l2": _plain(boosted.valid_score),
        "vol_floor": target_models.VOL_FLOOR,
        "market_bins": _edges(bins),
        "oos_mae": _mean(np.abs(error)),
        "importance_gain": _gains(boosted.booster, features),
    }
    run = _run(job, ModelObjective.VOLATILITY, VOL_DAYS, VOL_LABEL, params)
    model = {"booster": boosted.booster, "market_bins": dict(bins.edges)}
    column = volatility(boosted, inputs[job.read.test], features)
    return FoldFit({f"{MODEL}_vol_h{VOL_DAYS}": column}, [(run, model)])


def _calibration(scored: pd.DataFrame, column: str, suffix: str) -> dict[str, float | None]:
    """Pooled OOS ECE and Brier score of one probability column against the outcome up."""
    table = reliability_table(scored, "up", column, target_models.ECE_BINS)
    return {
        f"oos_ece{suffix}": _plain(expected_calibration_error(table)),
        f"oos_brier{suffix}": _mean(((scored[column] - scored["up"]) ** 2).to_numpy()),
    }


def _probability(job: Job, inputs: pd.DataFrame, labels: pd.DataFrame, bins: MarketBins) -> FoldFit:
    fold, features = job.fold, job.features
    fitted = fit_probability(inputs, labels, fold, features, job.seed, MAX_ROUNDS, job.bags)
    booster = fitted.boosted.booster
    evaluated = _evaluated(job, inputs, labels, UP_LABEL)
    scored = evaluated[["date"]].assign(
        up=evaluated[UP_LABEL].to_numpy(np.float64),
        prob=fitted.predict(evaluated, features),
        raw=np.asarray(booster.predict(evaluated[features])),
    )
    params = {
        "rounds": fitted.boosted.rounds,
        "valid_logloss": _plain(fitted.boosted.valid_score),
        "valid_ece": _plain(fitted.ece),
        "ece_limit": target_models.ECE_LIMIT,
        "calibrated": fitted.calibrator is not None,
        "market_bins": _edges(bins),
        **_calibration(scored, "prob", ""),
        **_calibration(scored, "raw", "_raw"),
        "importance_gain": _gains(booster, features),
    }
    run = _run(job, ModelObjective.CLASSIFICATION, VOL_DAYS, UP_LABEL, params)
    model = {"booster": booster, "calibrator": fitted.calibrator, "market_bins": dict(bins.edges)}
    test = inputs[job.read.test]
    columns = {
        f"{MODEL}_prob_h{VOL_DAYS}": fitted.predict(test, features),
        f"{MODEL}_prob_raw_h{VOL_DAYS}": np.asarray(booster.predict(test[features])),
    }
    return FoldFit(columns, [(run, model)])


def _interval(job: Job) -> FoldFit:
    inputs, bins = raw_inputs(job.read.raw, job.fold, job.features)
    labels = load_labels(job.fold.train_start, job.fold.eval_end, VOL_DAYS)
    parts = [fit(job, inputs, labels, bins) for fit in (_quantile, _volatility, _probability)]
    columns = {name: values for part in parts for name, values in part.columns.items()}
    return FoldFit(columns, [pair for part in parts for pair in part.trained])


FITS: dict[str, Callable[[Job], FoldFit]] = {
    HoldoutScope.RANK: _rank,
    HoldoutScope.RETURN: _return,
    HoldoutScope.INTERVAL: _interval,
}


def run_fold(
    scope: str,
    fold: Fold,
    run_key: str,
    directory: Path,
    features: list[str],
    seed: int,
    bags: int = 1,
    variant: Variant = SPEC_RECIPE,
    target: TargetVariant = TARGET_RECIPE,
) -> Path:
    """Fit one fold of a scope's boosters; the old cache goes first, the new one last."""
    if is_holdout(fold):
        raise HoldoutError(f"fold {fold.index} is holdout; it runs once with the frozen design")
    require_candidates(run_key, fold)
    fold_path(run_key, MODEL, fold.index).unlink(missing_ok=True)
    labels_digest = label_digest(scope, fold)
    if scope == HoldoutScope.RANK:
        read = rank_fold_inputs(fold, directory, features)
    else:
        read = fold_inputs(fold, fold.train_start, fold.test_end, directory, features)
    fitted = FITS[scope](Job(fold, run_key, read, features, seed, bags, variant))
    if label_digest(scope, fold) != labels_digest:
        raise RuntimeError(f"label_daily changed while fitting fold {fold.index}; run it again")
    for run, model in fitted.trained:
        record_run(run, model, fold=fold)
    rows = read.raw.loc[read.test, ROW_KEYS].reset_index(drop=True)
    notes = {
        "feature_hash": read.feature_hash,
        "label_hash": labels_digest,
        "recipe": recipe(scope, features, seed, bags, variant, target),
    }
    return save_fold_predictions(run_key, MODEL, fold, rows.assign(**fitted.columns), notes=notes)


def _stale(
    scope: str,
    run_key: str,
    fold: Fold,
    directory: Path,
    features: list[str],
    seed: int,
    bags: int,
    variant: Variant,
    target: TargetVariant,
) -> bool:
    """Whether a cached fold's features, labels or recipe have changed since it was fitted."""
    current = _current_notes(scope, fold, directory, features, seed, bags, variant, target)
    return fold_notes(run_key, MODEL, fold.index) != current


def run_folds(
    scope: str,
    folds: Iterable[Fold],
    run_key: str,
    directory: Path,
    seed: int = 0,
    rerun: bool = False,
    bags: int = 1,
    variant: Variant = SPEC_RECIPE,
    target: TargetVariant = TARGET_RECIPE,
) -> list[int]:
    """Run a scope's folds without a current cache, or all with rerun; holdout is refused."""
    check_run_key(scope, run_key)
    if scope != HoldoutScope.RANK and variant != SPEC_RECIPE:
        raise ValueError(f"design variants apply to the rank scope only, not {scope}")
    if scope == HoldoutScope.RANK and target != TARGET_RECIPE:
        raise ValueError("target variants apply to the return and interval scopes, not rank")
    chosen = list(folds)
    held = [fold.index for fold in chosen if is_holdout(fold)]
    if held:
        raise HoldoutError(f"folds {held} are holdout; they run once with the frozen design")
    bag_seeds(seed, bags)
    features = scope_features(scope, target)
    missing = set(missing_folds(run_key, MODEL, chosen))
    ran = []
    for fold in chosen:
        if (
            rerun
            or fold.index in missing
            or _stale(scope, run_key, fold, directory, features, seed, bags, variant, target)
        ):
            run_fold(scope, fold, run_key, directory, features, seed, bags, variant, target)
            ran.append(fold.index)
    return ran
