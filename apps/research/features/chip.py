"""Chip features of SPEC §6: institutional flows, margin, lending and TDCC holding changes."""

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from types import MappingProxyType

import numpy as np
import pandas as pd
from django.db.models import Min, Q

from apps.etl.runs import covered_days
from apps.market.calendar import TradingDays
from apps.market.models import DailyChip, Market, WeeklyHolding
from apps.research.features.common import (
    LISTING,
    carried,
    feature_frame,
    ratio,
    rolling,
    shifted,
    traded,
)
from apps.research.features.fundamental import sourced_name
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
LENDING: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "margin_balance": ("margin_balance", "margin_limit", "short_balance"),
        "sbl_short_balance": ("sbl_short_balance",),
    }
)
SOURCED_CHIPS = (
    "chip_margin_chg_5d",
    "chip_margin_usage",
    "chip_short_ratio",
    "chip_sbl_short_chg_5d",
)
LENDING_CELLS = [cell for cells in LENDING.values() for cell in cells]
EXCHANGE_TABLES: Mapping[str, str] = MappingProxyType(
    {"margin_balance": "margin", "sbl_short_balance": "sbl"}
)
EXCHANGES: Mapping[str, str] = MappingProxyType({Market.TWSE: "twse", Market.TPEX: "tpex"})
WRITTEN_COLUMNS = ["market", "source", "date"]
FILLER = 1.0
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


def _lending(panel: pd.DataFrame) -> dict[str, pd.Series]:
    margin = _balance(panel, "margin_balance")
    return {
        "chip_margin_chg_5d": _change(panel, "margin_balance"),
        "chip_margin_usage": ratio(margin, _balance(panel, "margin_limit")),
        "chip_short_ratio": ratio(_balance(panel, "short_balance"), margin),
        "chip_sbl_short_chg_5d": _change(panel, "sbl_short_balance"),
    }


def lending_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Margin and lending features of every row, as chip_features gives them."""
    require_sorted(panel)
    return feature_frame(panel, _lending(panel), SOURCED_CHIPS)


def chip_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Chip features of every row; needs the chip columns and the weekly holding changes."""
    require_sorted(panel)
    columns = {
        **{f"chip_fi_net_{days}d": _net(panel, "fi_net", days) for days in FLOW_DAYS},
        "chip_it_net_5d": _net(panel, "it_net", CHANGE_DAYS),
        "chip_dealer_self_net_5d": _net(panel, "dealer_self_net", CHANGE_DAYS),
        "chip_dealer_hedge_net_5d": _net(panel, "dealer_hedge_net", CHANGE_DAYS),
        **_lending(panel),
        "chip_fi_holding_pct": _balance(panel, "fi_holding_pct"),
        "chip_big_holder_pct_chg_1w": panel["big_holder_pct_chg"],
        "chip_holder_count_chg_1w": panel["holder_count_chg"],
    }
    return feature_frame(panel, columns, feature_names(Group.CHIP))


def _day_trades(panel: pd.DataFrame) -> pd.Series:
    """Day-trade shares; a stock left out of its market's table on a loaded day was not eligible."""
    shares = _flow(panel, "day_trade_volume")
    listed = panel["day_trade_volume"].notna()
    loaded = listed.groupby([panel["date"], panel["market"]]).transform("any")
    return shares.mask(loaded & shares.isna(), 0.0)


def _day_trade_ratio(panel: pd.DataFrame) -> pd.Series:
    """Day-trade share of volume over five days settled at least two trading days ago (T+2)."""
    trades = rolling(_day_trades(panel), panel, CHANGE_DAYS, "sum", CHANGE_DAYS)
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


def load_chip_starts() -> pd.DataFrame:
    """First day each stock has a margin and a lending balance on record, over the whole table."""
    firsts = {marker: Min("date", filter=Q(**{f"{marker}__isnull": False})) for marker in LENDING}
    rows = DailyChip.objects.values("stock_id").annotate(**firsts)
    frame = pd.DataFrame(list(rows), columns=["stock_id", *LENDING])
    return frame.set_index("stock_id")


def load_exchange_days(days: Sequence[date]) -> pd.DataFrame:
    """Market, source balance and day of each exchange margin or lending table a run wrote."""
    rows = [
        (market, marker, day)
        for market, source in EXCHANGES.items()
        for marker, table in EXCHANGE_TABLES.items()
        for day in sorted(
            covered_days(source, f"fill:{table}", days) | covered_days(source, table, days)
        )
    ]
    return pd.DataFrame(rows, columns=WRITTEN_COLUMNS)


def _both_written(written: pd.DataFrame, marker: str) -> list[date]:
    """Days whose table of the source both markets' runs wrote."""
    markets = written[written["source"] == marker].groupby("date")["market"].nunique()
    return list(markets[markets.eq(len(EXCHANGES))].index)


def chip_sourcing(panel: pd.DataFrame, starts: pd.DataFrame, written: pd.DataFrame) -> pd.DataFrame:
    """Per row, 0 where a lending feature is missing only for rows or shares the stock should have.

    A row is expected on a traded day whose tables not both markets wrote, from the stock's first
    row on or every day for a stock without any; halted days carry a row as the features do.
    """
    inputs = panel[[*LISTING, "date", "has_bar", "shares", *LENDING_CELLS]]
    days = pd.to_datetime(inputs["date"])
    priced = inputs["has_bar"].astype(bool)
    filled = inputs.assign(shares=inputs["shares"].fillna(FILLER))
    for marker, cells in LENDING.items():
        first = pd.to_datetime(inputs["stock_id"].map(starts[marker]))
        listed = inputs["date"].isin(_both_written(written, marker))
        expected = priced & ~listed & (first.isna() | days.ge(first))
        missing = inputs[marker].isna() & expected
        filled = filled.assign(**{cell: filled[cell].mask(missing, FILLER) for cell in cells})
    actual, whole = lending_features(inputs), lending_features(filled)
    flags = {
        sourced_name(name): (~(actual[name].isna() & whole[name].notna())).astype("float64")
        for name in SOURCED_CHIPS
    }
    return inputs[["stock_id", "date"]].assign(**flags)


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
