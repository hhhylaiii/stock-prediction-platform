"""Per-listing window helpers shared by the feature groups; panels are sorted by stock and date."""

import math
from collections.abc import Mapping, Sequence

import pandas as pd

ANNUAL_ROOT = math.sqrt(252)
LISTING = ["stock_id", "segment"]


def _keys(panel: pd.DataFrame) -> list[pd.Series]:
    return [panel[column] for column in LISTING]


def shifted(values: pd.Series, panel: pd.DataFrame, steps: int) -> pd.Series:
    return values.groupby(_keys(panel), sort=False).shift(steps)


def rolling(
    values: pd.Series, panel: pd.DataFrame, window: int, stat: str, min_periods: int
) -> pd.Series:
    """Rolling statistic over the last window rows of the listing; NaN values are skipped."""
    grouped = values.groupby(_keys(panel), sort=False).rolling(window, min_periods=min_periods)
    result: pd.Series = grouped.agg(stat)
    return result.droplevel(list(range(len(LISTING)))).reindex(values.index)


def ewm_mean(values: pd.Series, panel: pd.DataFrame, alpha: float) -> pd.Series:
    """Recursive mean seeded at the listing's first value in the panel; NaN inputs are skipped."""
    grouped = values.groupby(_keys(panel), sort=False)
    result: pd.Series = grouped.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    return result.droplevel(list(range(len(LISTING)))).reindex(values.index)


def filled_close(panel: pd.DataFrame) -> pd.Series:
    """Adjusted close carried over halted days within the listing."""
    return panel["adj_close"].groupby(_keys(panel), sort=False).ffill()


def flat_returns(close: pd.Series, panel: pd.DataFrame) -> pd.Series:
    """Daily returns of the carried close, so halted days count as 0 (same as fwd_vol_20d)."""
    return close / shifted(close, panel, 1) - 1


def closes_seen(close: pd.Series, panel: pd.DataFrame) -> pd.Series:
    return close.notna().groupby(_keys(panel), sort=False).cumsum()


def traded(panel: pd.DataFrame, column: str) -> pd.Series:
    """Volume or turnover with halted days as 0."""
    return panel[column].where(panel["has_bar"].astype(bool), 0.0).fillna(0.0)


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator != 0)


def feature_frame(
    panel: pd.DataFrame, columns: Mapping[str, pd.Series], names: Sequence[str]
) -> pd.DataFrame:
    """Stock, date and the named feature columns as float64, on the panel's index."""
    values = {name: columns[name].astype("float64") for name in names}
    return pd.DataFrame({"stock_id": panel["stock_id"], "date": panel["date"], **values})
