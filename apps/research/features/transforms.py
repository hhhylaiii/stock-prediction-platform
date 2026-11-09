"""Per-day transforms of SPEC §6 處理步驟 3–5, with parameters from the day's universe only."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.typing import DataFrameGroupBy

from apps.research.features.registry import FEATURES

LOW, HIGH = 0.01, 0.99
BIN_QUANTILES = (0.2, 0.4, 0.6, 0.8)
STOCK_LEVEL = [
    spec.name
    for spec in FEATURES
    if spec.batch == 1 and not spec.market_level and not spec.categorical
]
SECOND_STOCK_LEVEL = [
    spec.name
    for spec in FEATURES
    if spec.batch == 2 and not spec.market_level and not spec.categorical
]
MARKET_FEATURES = [spec.name for spec in FEATURES if spec.market_level]


def _universe_days(frame: pd.DataFrame, columns: Sequence[str]) -> DataFrameGroupBy:
    """The universe rows of the columns, grouped by day."""
    inside = frame[list(columns)].where(frame["in_universe"].astype(bool), axis=0)
    return inside.groupby(frame["date"])


def _per_row(daily: pd.DataFrame, name: str, frame: pd.DataFrame) -> np.ndarray:
    """One column of a per-day table, repeated on every row of its day."""
    return daily[name].reindex(frame["date"]).to_numpy(dtype=np.float64)


def winsorize(frame: pd.DataFrame, columns: Sequence[str] = STOCK_LEVEL) -> pd.DataFrame:
    """New frame clipped at the day's 1% and 99% universe quantiles.

    A bound that would flatten every universe value falls back to the universe extreme instead.
    """
    days = _universe_days(frame, columns)
    low, high = days.quantile(LOW), days.quantile(HIGH)
    smallest, largest = days.min(), days.max()
    lower, upper = low.where(low < largest, smallest), high.where(high > smallest, largest)
    return frame.assign(
        **{
            name: frame[name].clip(
                pd.Series(_per_row(lower, name, frame), index=frame.index),
                pd.Series(_per_row(upper, name, frame), index=frame.index),
            )
            for name in columns
        }
    )


def _scored(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(x − mean) / std, 0 where the day has no spread, NaN where x or the mean is missing."""
    with np.errstate(invalid="ignore"):
        scored = np.where(np.isnan(scale), 0.0, (values - center) / scale)
    return np.where(np.isnan(values) | np.isnan(center), np.nan, scored)


def zscore(frame: pd.DataFrame, columns: Sequence[str] = STOCK_LEVEL) -> pd.DataFrame:
    """New frame scored by the universe mean and sample std; a day without spread scores 0."""
    days = _universe_days(frame, columns)
    mean, std = days.mean(), days.std()
    scale = std.where(std > 0)
    return frame.assign(
        **{
            name: pd.Series(
                _scored(
                    frame[name].to_numpy(dtype=np.float64),
                    _per_row(mean, name, frame),
                    _per_row(scale, name, frame),
                ),
                index=frame.index,
            )
            for name in columns
        }
    )


def standardize(frame: pd.DataFrame, columns: Sequence[str] = STOCK_LEVEL) -> pd.DataFrame:
    """Winsorize then z-score, both on the day's universe."""
    return zscore(winsorize(frame, columns), columns)


def fill_with_flags(frame: pd.DataFrame, columns: Sequence[str] = STOCK_LEVEL) -> pd.DataFrame:
    """Fill gaps of z-scored columns with the day's universe median (0 if none) and flag them."""
    median = _universe_days(frame, columns).median().fillna(0.0)
    filled = frame.assign(
        **{
            name: frame[name].fillna(pd.Series(_per_row(median, name, frame), index=frame.index))
            for name in columns
        }
    )
    flags = {f"{name}_missing": frame[name].isna().astype(np.int8) for name in columns}
    return pd.concat([filled, pd.DataFrame(flags, index=frame.index)], axis=1)


@dataclass(frozen=True)
class MarketBins:
    """Quintile edges of market features fitted on a training window."""

    edges: Mapping[str, tuple[float, ...]]

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """New frame with each market feature replaced by its bin 0–4; NaN stays NaN."""
        binned = {
            name: pd.Series(
                np.searchsorted(np.asarray(edges), frame[name].to_numpy(), side="left"),
                index=frame.index,
                dtype="float64",
            ).where(frame[name].notna())
            for name, edges in self.edges.items()
        }
        return frame.assign(**binned)


def market_bins(train: pd.DataFrame, columns: Sequence[str]) -> MarketBins:
    """Edges from one value per training day, so the number of stocks never weights a day."""
    daily = train.drop_duplicates("date").set_index("date")[list(columns)]
    return MarketBins(
        {name: tuple(float(daily[name].quantile(q)) for q in BIN_QUANTILES) for name in columns}
    )
