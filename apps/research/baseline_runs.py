"""Walk-forward baseline folds: model files, model_run rows and cached predictions (SPEC §7)."""

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apps.research.baselines import (
    QUANTILES,
    LinearFit,
    fit_logistic,
    fit_ridge,
    historical_quantiles,
    shuffled_oos_ic,
    trailing_mean_return,
    vol_baseline,
    vol_interval,
    zero_return,
)
from apps.research.features.store import read_partitions
from apps.research.holdout import HoldoutError, is_holdout, require_key
from apps.research.labels import HORIZONS, VOL_DAYS
from apps.research.metrics.ic import daily_rank_ic, summarize_ic
from apps.research.models import HoldoutScope, ModelObjective, ModelRun, ModelStatus
from apps.research.prediction_cache import (
    fold_notes,
    fold_path,
    missing_folds,
    save_fold_predictions,
)
from apps.research.runs import (
    RunError,
    feature_hash,
    label_hash,
    lib_versions,
    record_run,
    screened_features,
)
from apps.research.training import (
    LINEAR_FEATURES,
    linear_features,
    linear_inputs,
    load_labels,
    training_rows,
)
from apps.research.walkforward import Fold

RECIPE_VERSION = "2"
BASELINE = "baseline"
RETURN_HORIZON = 5
MAX_RUN_KEY = 40
SHUFFLE_HORIZON = 20
RANK_LABEL = "fwd_rank_pct"
ROW_KEYS = ["date", "stock_id", "in_universe"]


@dataclass(frozen=True)
class FoldInputs:
    """Raw features a fold reads, with their linear inputs computed on first use."""

    fold: Fold
    raw: pd.DataFrame
    feature_hash: str
    features: list[str]

    @cached_property
    def inputs(self) -> pd.DataFrame:
        return linear_inputs(self.raw)

    def window(self, start: date, end: date) -> pd.Series:
        return self.raw["date"].between(start, end)

    def train(self, labels: pd.DataFrame, label: str) -> pd.DataFrame:
        """Training rows dated in the fold's training window."""
        inside = self.window(self.fold.train_start, self.fold.train_end)
        return training_rows(self.inputs[inside], labels, label)

    def evaluated(self, labels: pd.DataFrame, label: str) -> pd.DataFrame:
        """Tradable universe rows of test_start..eval_end with a known label."""
        inside = self.window(self.fold.test_start, self.fold.eval_end)
        return training_rows(self.inputs[inside], labels, label)

    @property
    def test(self) -> pd.Series:
        return self.window(self.fold.test_start, self.fold.test_end)


@dataclass(frozen=True)
class Trained:
    run: ModelRun
    model: Any | None


@dataclass(frozen=True)
class FoldResult:
    columns: dict[str, np.ndarray]
    trained: list[Trained]


def _plain(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _run(
    inputs: FoldInputs, run_key: str, model: str, horizon: int, label: str, **fields: Any
) -> ModelRun:
    fold = inputs.fold
    params = {"horizon": horizon, "label": label}
    return ModelRun(
        run_id=f"{run_key}.{model}.h{horizon}.f{fold.index:03d}",
        run_key=run_key,
        model_type=model,
        version=RECIPE_VERSION,
        train_start=fold.train_start,
        train_end=fold.train_end,
        lib_versions=lib_versions(),
        feature_hash=inputs.feature_hash,
        **{**fields, "params": {**params, **fields.get("params", {})}},
    )


def _linear(fit: LinearFit) -> dict[str, object]:
    scores = {str(penalty): _plain(score) for penalty, score in fit.scores.items()}
    return {"penalty": fit.penalty, "cv_scores": scores}


def _linear_params(fit: LinearFit) -> dict[str, object]:
    weights = fit.coefficients().set_index("feature")["weight"]
    return {**_linear(fit), "coefficients": {name: float(weights[name]) for name in fit.features}}


def _oos(fit: LinearFit, evaluated: pd.DataFrame, horizon: int) -> dict[str, float | None]:
    """The fold's OOS daily rank IC mean, ICIR and Newey-West t; empty without rows."""
    if evaluated.empty:
        return {"ic_mean": None, "icir": None, "ic_nw_t": None}
    daily = daily_rank_ic(evaluated.assign(_score=fit.predict(evaluated)), "_score", RANK_LABEL)
    summary = summarize_ic(daily, lag=horizon)
    return {
        "ic_mean": _plain(summary.mean),
        "icir": _plain(summary.icir),
        "ic_nw_t": _plain(summary.nw_t),
    }


def _rank(inputs: FoldInputs, run_key: str) -> FoldResult:
    fold, test = inputs.fold, inputs.inputs[inputs.test]
    columns, trained = {}, []
    for horizon in HORIZONS:
        labels = load_labels(fold.train_start, fold.eval_end, horizon)
        fit = fit_ridge(inputs.train(labels, RANK_LABEL), inputs.features, RANK_LABEL, horizon)
        columns[f"ridge_h{horizon}"] = fit.predict(test)
        oos = _oos(fit, inputs.evaluated(labels, RANK_LABEL), horizon)
        run = _run(
            inputs,
            run_key,
            "ridge",
            horizon,
            RANK_LABEL,
            objective=ModelObjective.RANK,
            params={**_linear_params(fit), "ic_nw_t": oos["ic_nw_t"]},
            feature_list=inputs.features,
            ic_mean=oos["ic_mean"],
            icir=oos["icir"],
        )
        trained.append(Trained(run, fit.model))
    return FoldResult(columns, trained)


def _return(inputs: FoldInputs, run_key: str) -> FoldResult:
    raw = inputs.raw[inputs.test]
    columns = {
        f"zero_h{RETURN_HORIZON}": zero_return(raw).to_numpy(),
        f"trailing_h{RETURN_HORIZON}": trailing_mean_return(raw, RETURN_HORIZON).to_numpy(),
    }
    return FoldResult(columns, [])


def _interval(inputs: FoldInputs, run_key: str) -> FoldResult:
    fold, raw, test = inputs.fold, inputs.raw[inputs.test], inputs.inputs[inputs.test]
    labels = load_labels(fold.train_start, fold.train_end, VOL_DAYS)
    up = inputs.train(labels, "fwd_up_5pct")
    fit = fit_logistic(up, inputs.features, "fwd_up_5pct", VOL_DAYS)
    quantiles = historical_quantiles(inputs.train(labels, "fwd_ret_open"), "fwd_ret_open")
    band = vol_interval(raw, VOL_DAYS)
    columns = {
        f"logistic_h{VOL_DAYS}": fit.predict(test),
        **{
            f"histq_{name}_h{VOL_DAYS}": np.full(len(raw), value)
            for name, value in quantiles.items()
        },
        **{f"volband_{name}_h{VOL_DAYS}": band[name].to_numpy() for name in QUANTILES},
        f"techvol_h{VOL_DAYS}": vol_baseline(raw).to_numpy(),
    }
    logistic = _run(
        inputs,
        run_key,
        "logistic",
        VOL_DAYS,
        "fwd_up_5pct",
        objective=ModelObjective.CLASSIFICATION,
        params=_linear_params(fit),
        feature_list=inputs.features,
    )
    histq = _run(
        inputs,
        run_key,
        "histq",
        VOL_DAYS,
        "fwd_ret_open",
        objective=ModelObjective.QUANTILE,
        params={"quantiles": quantiles},
        feature_list=[],
    )
    return FoldResult(columns, [Trained(logistic, fit.model), Trained(histq, None)])


SCOPES: dict[str, Callable[[FoldInputs, str], FoldResult]] = {
    HoldoutScope.RANK: _rank,
    HoldoutScope.RETURN: _return,
    HoldoutScope.INTERVAL: _interval,
}


def first_day(scope: str, fold: Fold) -> date:
    """First feature day a scope reads: the test period alone for the rule-only return scope."""
    return fold.test_start if scope == HoldoutScope.RETURN else fold.train_start


def label_digest(scope: str, fold: Fold) -> str:
    """label_hash of the rows a scope fits and evaluates on; empty for the rule-only scope."""
    if scope == HoldoutScope.RETURN:
        return ""
    if scope == HoldoutScope.RANK:
        return label_hash(fold.train_start, fold.eval_end, HORIZONS)
    return label_hash(fold.train_start, fold.train_end, (VOL_DAYS,))


def scope_features(scope: str) -> list[str]:
    """Linear inputs a scope fits on: the screened features and their gap flags; none for return."""
    if scope == HoldoutScope.RETURN:
        return []
    try:
        kept = screened_features().kept
    except RunError as error:
        raise RunError(f"the {scope} scope fits on the screened features: {error}") from error
    return linear_features(kept)


def recipe(features: list[str]) -> str:
    """RECIPE_VERSION with a digest of the features the linear baselines read."""
    listed = json.dumps(features).encode()
    return f"{RECIPE_VERSION}-{hashlib.sha256(listed).hexdigest()[:12]}"


def _current_notes(scope: str, fold: Fold, directory: Path, features: list[str]) -> dict[str, str]:
    """Notes a fold cached now would carry: its features, labels and recipe."""
    return {
        "feature_hash": feature_hash(first_day(scope, fold), fold.test_end, directory),
        "label_hash": label_digest(scope, fold),
        "recipe": recipe(features),
    }


def fold_inputs(
    fold: Fold, start: date, end: date, directory: Path, features: list[str]
) -> FoldInputs:
    """Raw features dated in [start, end], hashed before and after the read."""
    digest = feature_hash(start, end, directory)
    raw = read_partitions(start, end, directory)
    if feature_hash(start, end, directory) != digest:
        raise RuntimeError("feature partitions changed while reading; rerun after build_features")
    return FoldInputs(fold, raw, digest, features)


def require_candidates(run_key: str, fold: Fold) -> None:
    """Refuse to refit a fold of a run that already has a model past candidate."""
    rows = ModelRun.objects.filter(run_key=run_key, run_id__endswith=f".f{fold.index:03d}")
    kept = sorted(rows.exclude(status=ModelStatus.CANDIDATE).values_list("run_id", flat=True))
    if kept:
        raise RunError(f"{kept} are past candidate; fold {fold.index} keeps its cache")


def run_fold(scope: str, fold: Fold, run_key: str, directory: Path, features: list[str]) -> Path:
    """Fit one fold's baselines; its old cache goes first and the new one is written last."""
    if is_holdout(fold):
        raise HoldoutError(f"fold {fold.index} is holdout; it runs once with the Phase 4 models")
    require_candidates(run_key, fold)
    fold_path(run_key, BASELINE, fold.index).unlink(missing_ok=True)
    labels = label_digest(scope, fold)
    inputs = fold_inputs(fold, first_day(scope, fold), fold.test_end, directory, features)
    result = SCOPES[scope](inputs, run_key)
    if label_digest(scope, fold) != labels:
        raise RuntimeError(f"label_daily changed while fitting fold {fold.index}; run it again")
    for item in result.trained:
        record_run(item.run, item.model, fold=fold)
    rows = inputs.raw.loc[inputs.test, ROW_KEYS].reset_index(drop=True)
    notes = {"feature_hash": inputs.feature_hash, "label_hash": labels, "recipe": recipe(features)}
    predictions = rows.assign(**result.columns)
    return save_fold_predictions(run_key, BASELINE, fold, predictions, notes=notes)


def check_run_key(scope: str, run_key: str) -> None:
    """A run_key names a known scope, starts with it and leaves room for its run_ids."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {list(SCOPES)}, not {scope!r}")
    require_key(run_key)
    if not run_key.startswith(f"{scope}-"):
        raise ValueError(f"a {scope} run_key must start with {scope}-, got {run_key!r}")
    if len(run_key) > MAX_RUN_KEY:
        raise ValueError(f"run_key must be at most {MAX_RUN_KEY} characters to fit its run_ids")


def _stale(scope: str, run_key: str, fold: Fold, directory: Path, features: list[str]) -> bool:
    """Whether a cached fold's features, labels or recipe have changed since it was fitted."""
    current = _current_notes(scope, fold, directory, features)
    return fold_notes(run_key, BASELINE, fold.index) != current


def run_folds(
    scope: str, folds: Iterable[Fold], run_key: str, directory: Path, rerun: bool = False
) -> list[int]:
    """Run folds without a current cache, or all with rerun; holdout folds are refused."""
    check_run_key(scope, run_key)
    chosen = list(folds)
    held = [fold.index for fold in chosen if is_holdout(fold)]
    if held:
        raise HoldoutError(f"folds {held} are holdout; they run once with the Phase 4 models")
    features = scope_features(scope)
    missing = set(missing_folds(run_key, BASELINE, chosen))
    ran = []
    for fold in chosen:
        if rerun or fold.index in missing or _stale(scope, run_key, fold, directory, features):
            run_fold(scope, fold, run_key, directory, features)
            ran.append(fold.index)
    return ran


def shuffle_check(fold: Fold, directory: Path, seed: int) -> float:
    """OOS rank IC of the 20-day Ridge refit on labels shuffled within dates (SPEC §8)."""
    inputs = fold_inputs(fold, fold.train_start, fold.eval_end, directory, LINEAR_FEATURES)
    labels = load_labels(fold.train_start, fold.eval_end, SHUFFLE_HORIZON)
    frame = training_rows(inputs.inputs, labels, RANK_LABEL)
    return shuffled_oos_ic(frame, fold, LINEAR_FEATURES, RANK_LABEL, SHUFFLE_HORIZON, seed)
