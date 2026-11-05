"""Chip features of SPEC §6: institutional flows, margin, lending and TDCC holding changes."""

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd

from apps.market.calendar import TradingDays
from apps.market.models import DailyChip, WeeklyHolding
from apps.research.features.common import (
    carried,
    feature_frame,
    ratio,
    rolling,
    shifted,
    traded,
)
from apps.research.features.registry import Group, feature_names
from apps.research.panel import require_sorted

FLOW_DAYS = (1, 5, 20)
CHANGE_DAYS = 5
LONG_DAYS = 20
SETTLE_DAYS = 2
BIG_HOLDER_LEVELS = range(12, 16)
HOLDER_LEVELS = range(1, 16)
HOLDING_COLUMNS = ["stock_id", "publish_date", "big_holder_pct_chg", "holder_count_chg"]
CHIP_COLUMNS = [
    "fi_net",
    "it_net",
    "dealer_self_net",
    "dealer_hedge_net",
    "margin_balance",
    "margin_limit",
    "short_balance",
    "sbl_short_balance",
    "fi_holding_pct",
    "day_trade_volume",
]
HOLDING_LOOKBACK = timedelta(days=42)
HOLDING_STALE = pd.Timedelta(21, unit="D")


def _flow(panel: pd.DataFrame, column: str) -> pd.Series:
    """Net buys with halted days as 0; a traded day without a chip row stays missing."""
    return panel[column].where(panel["has_bar"].astype(bool), 0.0)


def _balance(panel: pd.DataFrame, column: str) -> pd.Series:
    """Balances with halted days carrying the last known value."""
    return panel[column].where(panel["has_bar"].astype(bool), carried(panel[column], panel))


def _net(panel: pd.DataFrame, column: str, days: int) -> pd.Series:
    return ratio(rolling(_flow(panel, column), panel, days, "sum", days), panel["shares"])


def _change(panel: pd.DataFrame, column: str) -> pd.Series:
    balance = _balance(panel, column)
    return ratio(balance - shifted(balance, panel, CHANGE_DAYS), panel["shares"])


def chip_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Chip features of every row; needs the chip columns and the weekly holding changes."""
    require_sorted(panel)
    margin = _balance(panel, "margin_balance")
    columns = {
        **{f"chip_fi_net_{days}d": _net(panel, "fi_net", days) for days in FLOW_DAYS},
        "chip_it_net_5d": _net(panel, "it_net", CHANGE_DAYS),
        "chip_dealer_self_net_5d": _net(panel, "dealer_self_net", CHANGE_DAYS),
        "chip_dealer_hedge_net_5d": _net(panel, "dealer_hedge_net", CHANGE_DAYS),
        "chip_margin_chg_5d": _change(panel, "margin_balance"),
        "chip_margin_usage": ratio(margin, _balance(panel, "margin_limit")),
        "chip_short_ratio": ratio(_balance(panel, "short_balance"), margin),
        "chip_fi_holding_pct": _balance(panel, "fi_holding_pct"),
        "chip_sbl_short_chg_5d": _change(panel, "sbl_short_balance"),
        "chip_big_holder_pct_chg_1w": panel["big_holder_pct_chg"],
        "chip_holder_count_chg_1w": panel["holder_count_chg"],
    }
    return feature_frame(panel, columns, feature_names(Group.CHIP))


def _day_trade_ratio(panel: pd.DataFrame) -> pd.Series:
    """Day-trade share of volume over five days settled at least two trading days ago (T+2)."""
    trades = rolling(_flow(panel, "day_trade_volume"), panel, CHANGE_DAYS, "sum", CHANGE_DAYS)
    volume = rolling(traded(panel, "volume"), panel, CHANGE_DAYS, "sum", CHANGE_DAYS)
    return shifted(ratio(trades, volume), panel, SETTLE_DAYS)


def _buy_streak(panel: pd.DataFrame) -> pd.Series:
    """Consecutive rows of foreign net buying up to each row; a missing flow resets the count."""
    flow = _flow(panel, "fi_net")
    buying = flow > 0
    runs = (~buying).groupby([panel["stock_id"], panel["segment"]]).cumsum()
    streak = buying.astype("float64").groupby([panel["stock_id"], panel["segment"], runs]).cumsum()
    return streak.where(flow.notna())


def second_batch_chip_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Second-batch chip features of every row; needs the chip columns with day-trade volume."""
    require_sorted(panel)
    holding = _balance(panel, "fi_holding_pct")
    columns = {
        "chip_day_trade_ratio_5d": _day_trade_ratio(panel),
        "chip_fi_holding_chg_20d": holding - shifted(holding, panel, LONG_DAYS),
        "chip_it_net_20d": _net(panel, "it_net", LONG_DAYS),
        "chip_fi_buy_streak": _buy_streak(panel),
    }
    return feature_frame(panel, columns, feature_names(Group.CHIP, batch=2))


def _trading_weeks(dates: pd.Series, calendar: TradingDays) -> pd.Series:
    """Number of weeks with a trading day whose Monday is on or before each date's Monday."""
    days = calendar.between(calendar.start, calendar.end)
    weeks = np.unique([day.toordinal() - day.weekday() for day in days])
    mondays = [day.toordinal() - day.weekday() for day in dates]
    return pd.Series(np.searchsorted(weeks, mondays, side="right"), index=dates.index)


def holding_changes(holdings: pd.DataFrame, calendar: TradingDays) -> pd.DataFrame:
    """Change of the 400-lot-and-up share (points) and of the holder count (ratio) per week.

    Missing unless the previous row is from the previous week with trading days.
    """
    levels = holdings["level"]
    big = holdings["pct"].where(levels.isin(BIG_HOLDER_LEVELS), 0.0)
    holders = holdings["holders"].where(levels.isin(HOLDER_LEVELS), 0)
    weekly = (
        holdings.assign(big=big, counted=holders)
        .groupby(["stock_id", "data_date", "publish_date"], as_index=False)[["big", "counted"]]
        .sum()
        .sort_values(["stock_id", "data_date"], ignore_index=True)
    )
    weeks = _trading_weeks(weekly["data_date"], calendar)
    adjacent = weeks - weeks.groupby(weekly["stock_id"]).shift(1) == 1
    previous = weekly.groupby("stock_id")[["big", "counted"]].shift(1).where(adjacent, axis=0)
    changes = weekly.assign(
        big_holder_pct_chg=weekly["big"] - previous["big"],
        holder_count_chg=ratio(weekly["counted"].astype("float64"), previous["counted"]) - 1,
    )
    return changes[HOLDING_COLUMNS]


def load_chips(start: date, end: date, stock_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """daily_chip rows dated in [start, end]."""
    rows = DailyChip.objects.filter(date__range=(start, end))
    if stock_ids is not None:
        rows = rows.filter(stock_id__in=list(stock_ids))
    columns = ["stock_id", "date", *CHIP_COLUMNS]
    return pd.DataFrame(list(rows.values_list(*columns)), columns=columns)


def attach_chips(panel: pd.DataFrame, chips: pd.DataFrame) -> pd.DataFrame:
    """New panel with the chip columns of each row's stock and day; days without a row are NaN."""
    lookup = chips.set_index(["stock_id", "date"])[CHIP_COLUMNS].astype("float64")
    keys = pd.MultiIndex.from_arrays([panel["stock_id"], panel["date"]])
    values = lookup.reindex(keys).to_numpy(dtype=np.float64)
    return panel.assign(**{column: values[:, k] for k, column in enumerate(CHIP_COLUMNS)})


def load_holding_changes(
    calendar: TradingDays, start: date, end: date, stock_ids: Sequence[str] | None = None
) -> pd.DataFrame:
    """Weekly holding changes published in [start − 42 days, end]."""
    rows = WeeklyHolding.objects.filter(publish_date__range=(start - HOLDING_LOOKBACK, end))
    if stock_ids is not None:
        rows = rows.filter(stock_id__in=list(stock_ids))
    columns = ["stock_id", "data_date", "publish_date", "level", "holders", "pct"]
    holdings = pd.DataFrame(list(rows.values_list(*columns)), columns=columns)
    if holdings.empty:
        return pd.DataFrame(columns=HOLDING_COLUMNS)
    return holding_changes(holdings, calendar)


def attach_holding_changes(panel: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """New panel with the latest change published on or before each day, at most 21 days old."""
    names = ["big_holder_pct_chg", "holder_count_chg"]
    if changes.empty or panel.empty:
        return panel.assign(**dict.fromkeys(names, np.nan))
    left = pd.DataFrame(
        {
            "row": np.arange(len(panel)),
            "stock_id": panel["stock_id"].to_numpy(),
            "on": pd.to_datetime(panel["date"]),
        }
    ).sort_values("on")
    right = changes.assign(on=pd.to_datetime(changes["publish_date"]))[["stock_id", "on", *names]]
    merged = pd.merge_asof(
        left, right.sort_values("on"), on="on", by="stock_id", tolerance=HOLDING_STALE
    ).sort_values("row")
    return panel.assign(**{name: merged[name].to_numpy(dtype=np.float64) for name in names})
