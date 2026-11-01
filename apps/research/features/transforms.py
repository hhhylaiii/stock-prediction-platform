"""Per-day transforms of SPEC §6 處理步驟 3–5, with parameters from the day's universe only."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from apps.research.features.registry import FEATURES

LOW, HIGH = 0.01, 0.99
BIN_QUANTILES = (0.2, 0.4, 0.6, 0.8)
STOCK_LEVEL = [
    spec.name
    for spec in FEATURES
    if spec.batch == 1 and not spec.market_level and not spec.categorical
]


def _by_day(
    frame: pd.DataFrame, columns: Sequence[str], stat: Callable[[DataFrameGroupBy], pd.DataFrame]
) -> pd.DataFrame:
    """A statistic of the universe rows of each day, repeated on every row of that day."""
    inside = frame[list(columns)].where(frame["in_universe"].astype(bool), axis=0)
    daily = stat(inside.groupby(frame["date"]))
    return daily.reindex(frame["date"]).set_axis(frame.index)


def winsorize(frame: pd.DataFrame, columns: Sequence[str] = STOCK_LEVEL) -> pd.DataFrame:
    """New frame with columns clipped at the 1% and 99% universe quantiles of the day."""
    low = _by_day(frame, columns, lambda days: days.quantile(LOW))
    high = _by_day(frame, columns, lambda days: days.quantile(HIGH))
    return frame.assign(**{name: frame[name].clip(low[name], high[name]) for name in columns})


def zscore(frame: pd.DataFrame, columns: Sequence[str] = STOCK_LEVEL) -> pd.DataFrame:
    """New frame with columns scored by the universe mean and sample std; no spread gives NaN."""
    mean = _by_day(frame, columns, lambda days: days.mean())
    std = _by_day(frame, columns, lambda days: days.std()).where(lambda values: values > 0)
    return frame.assign(**{name: (frame[name] - mean[name]) / std[name] for name in columns})


def standardize(frame: pd.DataFrame, columns: Sequence[str] = STOCK_LEVEL) -> pd.DataFrame:
    """Winsorize then z-score, both on the day's universe."""
    return zscore(winsorize(frame, columns), columns)


def fill_with_flags(frame: pd.DataFrame, columns: Sequence[str] = STOCK_LEVEL) -> pd.DataFrame:
    """New frame with gaps set to the universe median of the day (0 if none) and _missing flags."""
    median = _by_day(frame, columns, lambda days: days.median()).fillna(0.0)
    filled = frame.assign(**{name: frame[name].fillna(median[name]) for name in columns})
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
