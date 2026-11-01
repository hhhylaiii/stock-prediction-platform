"""Database inputs of the feature groups beyond the universe panel: shares and index returns."""

from collections.abc import Mapping, Sequence
from datetime import date
from types import MappingProxyType

import numpy as np
import pandas as pd

from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.market.models import IndexDaily, IndexId, Market, SharesOutstanding
from apps.research.features.common import FILL_LIMIT
from apps.research.features.size_risk import BETA_DAYS
from apps.research.features.technical import MOMENTUM_DAYS, YEAR_DAYS
from apps.research.universe import load_universe_panel

FEATURE_WARMUP = max(MOMENTUM_DAYS, YEAR_DAYS, BETA_DAYS) + FILL_LIMIT + 1
INDEX_OF_MARKET: Mapping[str, str] = MappingProxyType(
    {Market.TWSE.value: IndexId.TAIEX.value, Market.TPEX.value: IndexId.TPEX.value}
)
SHARE_COLUMNS = ["stock_id", "effective_date", "shares"]
RETURN_COLUMNS = ["index_id", "date", "market_ret"]


def load_shares(end: date, stock_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """Every shares_outstanding row effective on or before end."""
    rows = SharesOutstanding.objects.filter(effective_date__lte=end)
    if stock_ids is not None:
        rows = rows.filter(stock_id__in=list(stock_ids))
    return pd.DataFrame(list(rows.values_list(*SHARE_COLUMNS)), columns=SHARE_COLUMNS)


def attach_shares(panel: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """New panel with the shares in effect on each row's date; NaN before the first row."""
    if shares.empty or panel.empty:
        return panel.assign(shares=np.nan)
    left = pd.DataFrame(
        {
            "row": np.arange(len(panel)),
            "stock_id": panel["stock_id"].to_numpy(),
            "on": pd.to_datetime(panel["date"]),
        }
    ).sort_values("on")
    right = pd.DataFrame(
        {
            "stock_id": shares["stock_id"].to_numpy(),
            "on": pd.to_datetime(shares["effective_date"]),
            "shares": shares["shares"].astype("float64").to_numpy(),
        }
    ).sort_values("on")
    merged = pd.merge_asof(left, right, on="on", by="stock_id", direction="backward")
    return panel.assign(shares=merged.sort_values("row")["shares"].to_numpy())


def load_index_returns(calendar: TradingDays, start: date, end: date) -> pd.DataFrame:
    """Daily TAIEX and TPEX returns on the trading days in [start, end]; gaps give NaN."""
    days = calendar.between(max(start, calendar.start), min(end, calendar.end))
    if not days:
        return pd.DataFrame(columns=RETURN_COLUMNS)
    try:
        first = calendar.shift(days[0], -1)
    except OutsideCalendarError:
        first = days[0]
    rows = IndexDaily.objects.filter(
        index_id__in=list(INDEX_OF_MARKET.values()), date__range=(first, days[-1])
    ).values_list("index_id", "date", "close")
    closes = pd.DataFrame(list(rows), columns=["index_id", "date", "close"])
    wide = (
        closes.assign(close=closes["close"].astype("float64"))
        .pivot(index="date", columns="index_id", values="close")
        .reindex(
            index=list(calendar.between(first, days[-1])), columns=list(INDEX_OF_MARKET.values())
        )
    )
    returns = (wide / wide.shift(1) - 1).loc[list(days)]
    long = returns.rename_axis(index="date").reset_index()
    melted = long.melt(id_vars="date", var_name="index_id", value_name="market_ret")
    return melted[RETURN_COLUMNS]


def attach_market_returns(panel: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """New panel with the return of each row's market index: TWSE to TAIEX, TPEx to TPEX."""
    lookup = returns.set_index(["index_id", "date"])["market_ret"]
    keys = pd.MultiIndex.from_arrays([panel["market"].map(INDEX_OF_MARKET), panel["date"]])
    return panel.assign(market_ret=lookup.reindex(keys).to_numpy(dtype=np.float64))


def load_feature_panel(
    calendar: TradingDays, start: date, end: date, stock_ids: Sequence[str] | None = None
) -> pd.DataFrame:
    """Universe panel from FEATURE_WARMUP days before start, with shares and index returns."""
    days = calendar.between(max(start, calendar.start), min(end, calendar.end))
    warm_start = days[0] if days else start
    try:
        warm_start = calendar.shift(warm_start, -FEATURE_WARMUP) if days else warm_start
    except OutsideCalendarError:
        warm_start = calendar.start
    panel = load_universe_panel(calendar, warm_start, end, stock_ids)
    with_shares = attach_shares(panel, load_shares(end, stock_ids))
    return attach_market_returns(with_shares, load_index_returns(calendar, warm_start, end))
