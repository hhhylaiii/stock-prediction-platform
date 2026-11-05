"""LightGBM fits that stop on a validation tail, then refit the whole window (SPEC §7)."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from apps.research.metrics.ic import MIN_STOCKS, daily_rank_ic
from apps.research.walkforward import Fold

BASE_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "lambda_l2": 1.0,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 100
FEVAL_EVERY = 10
MAX_ROUNDS = 5000
RANK_IC = "daily_rank_ic"

Feval = Callable[[np.ndarray, Any], tuple[str, float, bool]]
FevalFactory = Callable[[pd.Series, pd.Series], Feval]


def seeded(seed: int) -> dict[str, int]:
    """LightGBM's row, bagging and feature-sampling seeds, distinct and derived from one seed."""
    return {"seed": seed, "bagging_seed": seed + 1, "feature_fraction_seed": seed + 2}


class DailyRankIC:
    """Validation feval: mean daily rank IC of the scores, recomputed every few rounds."""

    def __init__(self, dates: pd.Series, label: pd.Series, every: int = FEVAL_EVERY) -> None:
        self.frame = pd.DataFrame({"date": dates.to_numpy(), "label": label.to_numpy()})
        if not self.frame.groupby("date").size().ge(MIN_STOCKS).any():
            raise ValueError(f"no validation day has {MIN_STOCKS} stocks for a rank IC")
        self.every, self.calls, self.last = every, 0, 0.0

    def value(self, scores: np.ndarray) -> float:
        """Mean over days of the Spearman IC; 0 when no day has spread in the scores."""
        ic = daily_rank_ic(self.frame.assign(score=scores), "score", "label")
        return float(ic.mean()) if not ic.empty else 0.0

    def __call__(self, scores: np.ndarray, dataset: Any) -> tuple[str, float, bool]:
        if self.calls % self.every == 0:
            self.last = self.value(scores)
        self.calls += 1
        return RANK_IC, self.last, True


def split_validation(rows: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit rows up to fit_end and validation rows from valid_start; none may pass train_end."""
    late = int((rows["date"] > fold.train_end).sum())
    if late:
        raise ValueError(f"{late} training rows fall after train_end {fold.train_end}")
    return rows[rows["date"] <= fold.fit_end], rows[rows["date"] >= fold.valid_start]


@dataclass(frozen=True)
class Boosted:
    booster: lgb.Booster
    rounds: int
    valid_score: float


def _dataset(
    rows: pd.DataFrame, features: Sequence[str], label: str, reference: lgb.Dataset | None = None
) -> lgb.Dataset:
    return lgb.Dataset(rows[list(features)], label=rows[label], reference=reference)


def boost(
    rows: pd.DataFrame,
    fold: Fold,
    features: Sequence[str],
    label: str,
    seed: int,
    params: Mapping[str, Any] | None = None,
    feval: FevalFactory | None = None,
    max_rounds: int = MAX_ROUNDS,
) -> Boosted:
    """Find the best round on the validation tail, then refit every row for that many rounds."""
    fit, valid = split_validation(rows, fold)
    settings = {**BASE_PARAMS, **(params or {}), **seeded(seed)}
    if feval is not None:
        settings["metric"] = "None"
    fit_set = _dataset(fit, features, label)
    probe = lgb.train(
        settings,
        fit_set,
        num_boost_round=max_rounds,
        valid_sets=[_dataset(valid, features, label, reference=fit_set)],
        valid_names=["valid"],
        feval=None if feval is None else feval(valid["date"], valid[label]),
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    rounds = probe.best_iteration or probe.current_iteration()
    score = next(iter(probe.best_score["valid"].values()))
    final = lgb.train(settings, _dataset(rows, features, label), num_boost_round=rounds)
    return Boosted(final, rounds, float(score))
