"""Walk-forward baseline folds: model files, model_run rows and cached predictions (SPEC §7)."""

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import date
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
from apps.research.holdout import HoldoutError, is_holdout
from apps.research.labels import HORIZONS, VOL_DAYS
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import HoldoutScope, ModelObjective, ModelRun
from apps.research.prediction_cache import missing_folds, save_fold_predictions
from apps.research.runs import feature_hash, lib_versions, record_run
from apps.research.training import LINEAR_FEATURES, linear_inputs, load_labels, training_rows
from apps.research.walkforward import Fold

RECIPE_VERSION = "1"
RETURN_HORIZON = 5
MAX_RUN_KEY = 40
SHUFFLE_HORIZON = 20
RANK_LABEL = "fwd_rank_pct"
ROW_KEYS = ["date", "stock_id", "in_universe"]


@dataclass(frozen=True)
class FoldInputs:
    """Raw features over train_start..test_end and their linear inputs, row for row."""

    fold: Fold
    raw: pd.DataFrame
    inputs: pd.DataFrame
    feature_hash: str

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


def _fold_record(fold: Fold) -> dict[str, object]:
    return {
        name: value.isoformat() if isinstance(value, date) else value
        for name, value in asdict(fold).items()
    }


def _run(
    inputs: FoldInputs, run_key: str, model: str, horizon: int, label: str, **fields: Any
) -> ModelRun:
    fold = inputs.fold
    params = {"horizon": horizon, "label": label, "fold": _fold_record(fold)}
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


def _ic(daily: pd.Series) -> tuple[float | None, float | None]:
    if daily.empty:
        return None, None
    mean, spread = float(daily.mean()), float(daily.std())
    return _plain(mean), _plain(mean / spread) if spread > 0 else None


def _rank(inputs: FoldInputs, run_key: str) -> FoldResult:
    fold, test = inputs.fold, inputs.inputs[inputs.test]
    columns, trained = {}, []
    for horizon in HORIZONS:
        labels = load_labels(fold.train_start, fold.eval_end, horizon)
        fit = fit_ridge(inputs.train(labels, RANK_LABEL), LINEAR_FEATURES, RANK_LABEL, horizon)
        columns[f"ridge_h{horizon}"] = fit.predict(test)
        evaluated = inputs.evaluated(labels, RANK_LABEL)
        ic_mean, icir = _ic(
            daily_rank_ic(evaluated.assign(_score=fit.predict(evaluated)), "_score", RANK_LABEL)
        )
        run = _run(
            inputs,
            run_key,
            "ridge",
            horizon,
            RANK_LABEL,
            objective=ModelObjective.RANK,
            params=_linear(fit),
            feature_list=LINEAR_FEATURES,
            ic_mean=ic_mean,
            icir=icir,
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
    fit = fit_logistic(up, LINEAR_FEATURES, "fwd_up_5pct", VOL_DAYS)
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
        params=_linear(fit),
        feature_list=LINEAR_FEATURES,
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


def fold_inputs(fold: Fold, end: date, directory: Path) -> FoldInputs:
    """Raw features from train_start through end, hashed before and after the read."""
    digest = feature_hash(fold.train_start, end, directory)
    raw = read_partitions(fold.train_start, end, directory)
    if feature_hash(fold.train_start, end, directory) != digest:
        raise RuntimeError("feature partitions changed while reading; rerun after build_features")
    return FoldInputs(fold, raw, linear_inputs(raw), digest)


def run_fold(scope: str, fold: Fold, run_key: str, directory: Path) -> Path:
    """Fit one fold's baselines; model files and rows first, the predictions cache last."""
    inputs = fold_inputs(fold, fold.test_end, directory)
    result = SCOPES[scope](inputs, run_key)
    for item in result.trained:
        record_run(item.run, item.model)
    rows = inputs.raw.loc[inputs.test, ROW_KEYS].reset_index(drop=True)
    return save_fold_predictions(run_key, fold, rows.assign(**result.columns))


def _check(scope: str, run_key: str) -> None:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {list(SCOPES)}, not {scope!r}")
    if not run_key.startswith(f"{scope}-"):
        raise ValueError(f"a {scope} run_key must start with {scope}-, got {run_key!r}")
    if len(run_key) > MAX_RUN_KEY:
        raise ValueError(f"run_key must be at most {MAX_RUN_KEY} characters to fit its run_ids")


def run_folds(
    scope: str, folds: Iterable[Fold], run_key: str, directory: Path, rerun: bool = False
) -> list[int]:
    """Run the folds without cached predictions, or all with rerun; holdout folds are refused."""
    _check(scope, run_key)
    chosen = list(folds)
    held = [fold.index for fold in chosen if is_holdout(fold)]
    if held:
        raise HoldoutError(f"folds {held} are holdout; they run once with the Phase 4 models")
    pending = {fold.index for fold in chosen} if rerun else set(missing_folds(run_key, chosen))
    ran = []
    for fold in chosen:
        if fold.index in pending:
            run_fold(scope, fold, run_key, directory)
            ran.append(fold.index)
    return ran


def shuffle_check(fold: Fold, directory: Path, seed: int) -> float:
    """OOS rank IC of the 20-day Ridge refit on labels shuffled within dates (SPEC §8)."""
    inputs = fold_inputs(fold, fold.eval_end, directory)
    labels = load_labels(fold.train_start, fold.eval_end, SHUFFLE_HORIZON)
    frame = training_rows(inputs.inputs, labels, RANK_LABEL)
    return shuffled_oos_ic(frame, fold, LINEAR_FEATURES, RANK_LABEL, SHUFFLE_HORIZON, seed)
