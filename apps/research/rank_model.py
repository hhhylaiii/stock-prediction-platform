"""LightGBM rank model on screened z-scored features, ranked within each day (SPEC §7)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from apps.research.features.transforms import STOCK_LEVEL, standardize
from apps.research.gbdt import MAX_ROUNDS, Boosted, DailyRankIC, Feval, Layout, boost
from apps.research.labels import HORIZONS
from apps.research.runs import Screened
from apps.research.training import training_rows
from apps.research.walkforward import Fold, time_decay_weights

RANK_LABEL = "fwd_rank_pct"
MODEL_TYPE = "lightgbm"
ROW_COLUMNS = ["date", "stock_id", "in_universe"]
OBJECTIVES = ("regression", "lambdarank")
LEVELS = 10
GRADE = "fwd_rank_level"
WEIGHT = "decay_weight"
LAMBDARANK = {
    "objective": "lambdarank",
    "lambdarank_truncation_level": 30,
    "label_gain": [2.0**level - 1 for level in range(LEVELS)],
}


@dataclass(frozen=True)
class Variant:
    """A rank design choice under comparison in research; the defaults are the SPEC recipe."""

    every: int = 1
    half_life: float | None = None
    objective: str = "regression"
    min_rounds: int = 0
    fixed_rounds: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f"every must be at least 1, not {self.every}")
        if self.half_life is not None and self.half_life <= 0:
            raise ValueError(f"half_life must be positive, not {self.half_life}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, not {self.objective!r}")
        if self.min_rounds < 0:
            raise ValueError(f"min_rounds must not be negative, not {self.min_rounds}")
        horizons = [horizon for horizon, _ in self.fixed_rounds]
        unknown = set(horizons) - set(HORIZONS)
        if (
            unknown
            or len(set(horizons)) < len(horizons)
            or any(n < 1 for _, n in self.fixed_rounds)
        ):
            raise ValueError(
                f"fixed_rounds needs one positive count per horizon: {self.fixed_rounds}"
            )

    def fixed_for(self, horizon: int) -> int:
        """The horizon's fixed refit rounds; 0 leaves them to early stopping."""
        return dict(self.fixed_rounds).get(horizon, 0)


SPEC_RECIPE = Variant()


def _stock_level(features: Sequence[str]) -> list[str]:
    unknown = [name for name in features if name not in STOCK_LEVEL]
    if unknown:
        raise ValueError(f"the rank model takes stock-level features only, not {unknown}")
    return list(features)


def rank_features(screened: Screened) -> list[str]:
    """The screened stock-level features; the rank model never sees market features."""
    return _stock_level(screened.kept)


def rank_inputs(raw: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Features winsorized and z-scored on each day's universe, gaps left for LightGBM."""
    return standardize(raw, features)


def fit_rank(
    inputs: pd.DataFrame,
    labels: pd.DataFrame,
    fold: Fold,
    features: Sequence[str],
    seed: int,
    max_rounds: int = MAX_ROUNDS,
    bags: int = 1,
    variant: Variant = SPEC_RECIPE,
    fixed_rounds: int = 0,
) -> Boosted:
    """Fit on the fold's tradable universe rows, stopping on validation daily rank IC.

    lambdarank learns LEVELS integer grades of the rank; every variant stops on the rank itself.
    """
    names = _stock_level(features)
    rows = training_rows(
        inputs[inputs["date"].between(fold.train_start, fold.train_end)], labels, RANK_LABEL
    )
    label, params, weight = RANK_LABEL, None, None
    if variant.half_life is not None:
        decay = time_decay_weights(rows["date"], fold.train_end, variant.half_life)
        rows, weight = rows.assign(**{WEIGHT: decay}), WEIGHT
    ranking = variant.objective == "lambdarank"
    if ranking:
        grades = np.minimum(np.floor(rows[RANK_LABEL] * LEVELS), LEVELS - 1).astype(np.int64)
        rows, label, params = rows.assign(**{GRADE: grades}), GRADE, LAMBDARANK
    pct = rows[RANK_LABEL]

    def feval(dates: pd.Series, _: pd.Series) -> Feval:
        return DailyRankIC(dates, pct.loc[dates.index])

    layout = Layout(every=variant.every, weight=weight, group=ranking)
    return boost(
        rows,
        fold,
        names,
        label,
        seed,
        params,
        feval,
        max_rounds,
        bags,
        layout,
        min_rounds=variant.min_rounds,
        fixed_rounds=fixed_rounds,
    )


def prediction_frame(rows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Scores with each day's universe rank (1 is the highest) and percentile; others unranked."""
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("rank scores must all be finite")
    frame = rows[ROW_COLUMNS].reset_index(drop=True).assign(score=values)
    inside = frame[frame["in_universe"].astype(bool)].sort_values(
        ["date", "score", "stock_id"], ascending=[True, False, True]
    )
    rank = inside.groupby("date").cumcount() + 1
    count = inside.groupby("date")["score"].transform("size")
    return frame.assign(
        rank=rank.reindex(frame.index).astype("Int64"),
        pct=(1 - (rank - 1) / count).reindex(frame.index),
    )
