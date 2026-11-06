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
BAG_STRIDE = 3
RANK_IC = "daily_rank_ic"

Feval = Callable[[np.ndarray, Any], tuple[str, float, bool]]
FevalFactory = Callable[[pd.Series, pd.Series], Feval]


def seeded(seed: int) -> dict[str, int]:
    """LightGBM's row, bagging and feature-sampling seeds, distinct and derived from one seed."""
    return {"seed": seed, "bagging_seed": seed + 1, "feature_fraction_seed": seed + 2}


def bag_seeds(seed: int, bags: int) -> list[int]:
    """Seeds of a bag, spaced so no two members share any of their three LightGBM seeds."""
    if bags < 1:
        raise ValueError(f"a bag needs at least one seed, not {bags}")
    return [seed + BAG_STRIDE * member for member in range(bags)]


@dataclass(frozen=True)
class Bagged:
    """Boosters of one recipe and round count under different seeds, averaged."""

    boosters: tuple[lgb.Booster, ...]

    def predict(self, data: Any) -> np.ndarray:
        return np.mean([np.asarray(member.predict(data)) for member in self.boosters], axis=0)

    def feature_importance(self, importance_type: str) -> np.ndarray:
        gains = [member.feature_importance(importance_type) for member in self.boosters]
        return np.mean(gains, axis=0)


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
    """The refit booster, its round count, and the probe's validation score and predictions."""

    booster: lgb.Booster | Bagged
    rounds: int
    valid_score: float
    valid_scores: np.ndarray


@dataclass(frozen=True)
class Layout:
    """How a fit reads its rows: every n-th day, a weight column, and day groups for ranking."""

    every: int = 1
    weight: str | None = None
    group: bool = False


EVERY_ROW = Layout()


def sampled(rows: pd.DataFrame, every: int) -> pd.DataFrame:
    """Rows of every n-th day counted back from the last day; rows itself when every is 1."""
    if every < 1:
        raise ValueError(f"every must be at least 1, not {every}")
    if every == 1:
        return rows
    days = pd.Index(sorted(rows["date"].unique()))
    return rows[rows["date"].isin(days[::-1][::every])]


def _dataset(
    rows: pd.DataFrame,
    features: Sequence[str],
    label: str,
    reference: lgb.Dataset | None = None,
    layout: Layout = EVERY_ROW,
    weighted: bool = True,
) -> lgb.Dataset:
    return lgb.Dataset(
        rows[list(features)],
        label=rows[label],
        reference=reference,
        weight=rows[layout.weight] if weighted and layout.weight else None,
        group=rows.groupby("date", sort=False).size().to_numpy() if layout.group else None,
    )


def refit(
    rows: pd.DataFrame,
    features: Sequence[str],
    label: str,
    seed: int,
    rounds: int,
    params: Mapping[str, Any] | None = None,
    layout: Layout = EVERY_ROW,
) -> lgb.Booster:
    """A booster trained on every row for exactly the given rounds."""
    settings = {**BASE_PARAMS, **(params or {}), **seeded(seed)}
    data = _dataset(rows, features, label, layout=layout)
    return lgb.train(settings, data, num_boost_round=rounds)


def boost(
    rows: pd.DataFrame,
    fold: Fold,
    features: Sequence[str],
    label: str,
    seed: int,
    params: Mapping[str, Any] | None = None,
    feval: FevalFactory | None = None,
    max_rounds: int = MAX_ROUNDS,
    bags: int = 1,
    layout: Layout = EVERY_ROW,
    min_rounds: int = 0,
) -> Boosted:
    """Find the best round on the validation tail, then refit every row per bag seed.

    Sampling thins the fit and refit rows, not the validation tail; min_rounds floors the refit.
    """
    fit, valid = split_validation(rows, fold)
    kept = sampled(rows, layout.every)
    fit = kept[kept["date"] <= fold.fit_end]
    settings = {**BASE_PARAMS, **(params or {}), **seeded(seed)}
    if feval is not None:
        settings["metric"] = "None"
    fit_set = _dataset(fit, features, label, layout=layout)
    probe = lgb.train(
        settings,
        fit_set,
        num_boost_round=max_rounds,
        valid_sets=[_dataset(valid, features, label, fit_set, layout, weighted=False)],
        valid_names=["valid"],
        feval=None if feval is None else feval(valid["date"], valid[label]),
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best = probe.best_iteration or probe.current_iteration()
    score = next(iter(probe.best_score["valid"].values()))
    scores = np.asarray(probe.predict(valid[list(features)], num_iteration=best))
    rounds = max(best, min_rounds)
    members = [
        refit(kept, features, label, member, rounds, params, layout)
        for member in bag_seeds(seed, bags)
    ]
    final = members[0] if bags == 1 else Bagged(tuple(members))
    return Boosted(final, rounds, float(score), scores)
