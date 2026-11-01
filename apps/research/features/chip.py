"""Chip features of SPEC §6: institutional flows, margin, lending and TDCC holding changes."""

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd

from apps.market.models import DailyChip, WeeklyHolding
from apps.research.features.common import feature_frame, ratio, rolling, shifted
from apps.research.features.registry import Group, feature_names
from apps.research.panel import require_sorted

FLOW_DAYS = (1, 5, 20)
CHANGE_DAYS = 5
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
]
HOLDING_LOOKBACK = timedelta(days=35)
HOLDING_STALE = pd.Timedelta(14, unit="D")


def _flow(panel: pd.DataFrame, column: str) -> pd.Series:
    """Net buys with halted days as 0; a traded day without a chip row stays missing."""
    return panel[column].where(panel["has_bar"].astype(bool), 0.0)


def _balance(panel: pd.DataFrame, column: str) -> pd.Series:
    """Balances with halted days carrying the last known value."""
    carried = panel[column].groupby([panel["stock_id"], panel["segment"]], sort=False).ffill()
    return panel[column].where(panel["has_bar"].astype(bool), carried)


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


def holding_changes(holdings: pd.DataFrame) -> pd.DataFrame:
    """Week-on-week change of the 400-lot-and-up share (points) and of the holder count (ratio)."""
    levels = holdings["level"]
    big = holdings["pct"].where(levels.isin(BIG_HOLDER_LEVELS), 0.0)
    holders = holdings["holders"].where(levels.isin(HOLDER_LEVELS), 0)
    weekly = (
        holdings.assign(big=big, counted=holders)
        .groupby(["stock_id", "data_date", "publish_date"], as_index=False)[["big", "counted"]]
        .sum()
        .sort_values(["stock_id", "data_date"], ignore_index=True)
    )
    previous = weekly.groupby("stock_id")[["big", "counted"]].shift(1)
    changes = weekly.assign(
        big_holder_pct_chg=weekly["big"] - previous["big"],
        holder_count_chg=ratio(weekly["counted"].astype("float64"), previous["counted"]) - 1,
    )
    return changes.loc[previous["big"].notna(), HOLDING_COLUMNS].reset_index(drop=True)


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
    start: date, end: date, stock_ids: Sequence[str] | None = None
) -> pd.DataFrame:
    """Weekly holding changes published in [start − 35 days, end]."""
    rows = WeeklyHolding.objects.filter(publish_date__range=(start - HOLDING_LOOKBACK, end))
    if stock_ids is not None:
        rows = rows.filter(stock_id__in=list(stock_ids))
    columns = ["stock_id", "data_date", "publish_date", "level", "holders", "pct"]
    holdings = pd.DataFrame(list(rows.values_list(*columns)), columns=columns)
    if holdings.empty:
        return pd.DataFrame(columns=HOLDING_COLUMNS)
    return holding_changes(holdings)


def attach_holding_changes(panel: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """New panel with the latest change published on or before each day, at most 14 days old."""
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
