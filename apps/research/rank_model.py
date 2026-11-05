"""LightGBM rank model on screened z-scored features, ranked within each day (SPEC §7)."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from apps.research.features.transforms import STOCK_LEVEL, standardize
from apps.research.gbdt import MAX_ROUNDS, Boosted, DailyRankIC, boost
from apps.research.runs import Screened
from apps.research.training import training_rows
from apps.research.walkforward import Fold

RANK_LABEL = "fwd_rank_pct"
MODEL_TYPE = "lightgbm"
ROW_COLUMNS = ["date", "stock_id", "in_universe"]


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
) -> Boosted:
    """Fit on the fold's tradable universe rows, stopping on validation daily rank IC."""
    names = _stock_level(features)
    window = inputs[inputs["date"].between(fold.train_start, fold.train_end)]
    rows = training_rows(window, labels, RANK_LABEL)
    return boost(
        rows, fold, names, RANK_LABEL, seed, feval=DailyRankIC, max_rounds=max_rounds, bags=bags
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
