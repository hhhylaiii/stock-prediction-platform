"""Decile portfolios held as overlapping tranches, net of the costs of what they trade (SPEC §7)."""

import math

import numpy as np
import numpy.typing as npt
import pandas as pd

from apps.market.calendar import TradingDays
from apps.research.metrics.stats import equal_count_buckets

GROUPS = 10
FEE = 0.001425
TAX = 0.003
BUY_COST = FEE
SELL_COST = FEE + TAX
ANNUAL_DAYS = 252
UNIVERSE = "universe"
LIQUIDITY = "liq_dollar_vol_med_20d"
DAILY_COLUMNS = ["date", "portfolio", "gross", "cost", "net", "turnover", "sleeves"]
SUMMARY_COLUMNS = [
    "monthly_return",
    "monthly_excess",
    "excess_drawdown",
    "annual_turnover",
    "n_days",
]
Matrix = npt.NDArray[np.float64]


def decile_groups(signal: pd.DataFrame, groups: int = GROUPS) -> pd.Series:
    """Group 1..groups of each row's score among its day's rows, equal counts; top scores last."""
    ordered = signal.sort_values(["date", "stock_id"])
    return equal_count_buckets(ordered["score"], ordered["date"], groups).reindex(signal.index)


def top_liquidity(frame: pd.DataFrame, by: str = LIQUIDITY, n_buckets: int = 3) -> pd.DataFrame:
    """Rows in the top of n_buckets equal-count buckets of by on each day."""
    rows = frame[frame[by].notna()]
    ordered = rows.sort_values(["date", "stock_id"])
    top = equal_count_buckets(ordered[by], ordered["date"], n_buckets) == n_buckets
    return rows[top.reindex(rows.index)]


def _signal_rows(signal: pd.DataFrame, calendar: TradingDays, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    rows = signal.loc[signal["score"].notna(), ["date", "stock_id", "score"]]
    if rows.duplicated(["date", "stock_id"]).any():
        raise ValueError("signal has more than one row per stock and day")
    off = sorted(day for day in set(rows["date"]) if not calendar.is_trading_day(day))
    if off:
        raise ValueError(f"signal date {off[0]} is not a trading day")
    return rows.reset_index(drop=True)


def _wide(prices: pd.DataFrame, column: str, days: pd.Index, stocks: pd.Index) -> Matrix:
    table = prices.pivot(index="date", columns="stock_id", values=column)
    return table.reindex(index=days, columns=stocks).to_numpy(dtype=np.float64)


def _entries(opens: Matrix, closes: Matrix, at: np.ndarray, col: np.ndarray) -> Matrix:
    """Next-day open of each signal row; refuses rows that cannot trade there."""
    following = at + 1
    inside = (at >= 0) & (following < len(opens))
    entry = np.full(len(at), np.nan)
    entry[inside] = opens[following[inside], col[inside]]
    settled = np.zeros(len(at), dtype=bool)
    settled[inside] = np.isfinite(closes[following[inside], col[inside]])
    if not (np.isfinite(entry) & (entry > 0) & settled).all():
        raise ValueError("a signal row has no open and close on the next trading day")
    return entry


def _values(weights: Matrix, closes: Matrix, horizon: int) -> Matrix:
    """Value of each day's tranche k days on, per unit bought at the next open."""
    days = len(weights)
    values = np.full((days, horizon + 1), np.nan)
    values[:, 0] = (weights > 0).any(axis=1)
    for k in range(1, horizon + 1):
        if k < days:
            values[: days - k, k] = (weights[: days - k] * closes[k:]).sum(axis=1)
    return values


def _trades(
    members: Matrix, weights: Matrix, closes: Matrix, values: Matrix, horizon: int
) -> tuple[Matrix, Matrix]:
    """Buys and sells of each sleeve roll: the exiting tranche's drifted weights to the new ones."""
    counts = members.sum(axis=1, keepdims=True)
    new = np.divide(members, counts, out=np.zeros_like(members), where=counts > 0)
    old = np.zeros_like(members)
    exit_value = values[:-horizon, horizon, None] if horizon < len(values) else np.empty((0, 1))
    if len(exit_value):
        held = weights[:-horizon] * closes[horizon:]
        old[horizon:] = np.divide(held, exit_value, out=np.zeros_like(held), where=exit_value > 0)
    change = new - old
    return np.clip(change, 0, None).sum(axis=1), np.clip(-change, 0, None).sum(axis=1)


def _portfolio(
    members: Matrix, entry: Matrix, closes: Matrix, horizon: int
) -> tuple[Matrix, Matrix, Matrix, npt.NDArray[np.int64]]:
    """Gross return, cost, turnover and live sleeves per day of one set of daily tranches."""
    counts = members.sum(axis=1, keepdims=True)
    weights = np.divide(members / entry, counts, out=np.zeros_like(members), where=counts > 0)
    values = _values(weights, closes, horizon)
    live = values[:, 0] > 0
    days = len(members)
    gross = np.zeros(days)
    sleeves = np.zeros(days, dtype=np.int64)
    for k in range(1, min(horizon, days - 1) + 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            step = values[: days - k, k] / values[: days - k, k - 1] - 1
        gross[k:] += np.where(live[: days - k], step, 0.0)
        sleeves[k:] += live[: days - k]
    buys, sells = _trades(members, weights, closes, values, horizon)
    bought = np.concatenate([[0.0], buys[:-1]])
    cost = (bought * BUY_COST + sells * SELL_COST) / horizon
    return gross / horizon, cost, (bought + sells) / (2 * horizon), sleeves


def tranche_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: TradingDays,
    horizon: int,
    groups: int = GROUPS,
) -> pd.DataFrame:
    """Daily returns of each score group and the universe held as overlapping horizon-day tranches.

    Rows must be tradable at the next open (untradable_t1 = 0); prices hold adj_open and adj_close.
    """
    rows = _signal_rows(signal, calendar, horizon)
    if rows.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    last = min(prices["date"].max(), calendar.end)
    days = pd.Index(calendar.between(rows["date"].min(), last))
    stocks = pd.Index(sorted(set(rows["stock_id"])))
    used = prices[prices["stock_id"].isin(stocks) & prices["date"].isin(days)]
    opens = _wide(used, "adj_open", days, stocks)
    raw = _wide(used, "adj_close", days, stocks)
    at = days.get_indexer(pd.Index(rows["date"]))
    col = stocks.get_indexer(pd.Index(rows["stock_id"]))
    entry = np.ones_like(opens)
    entry[at, col] = _entries(opens, raw, at, col)
    closes = np.nan_to_num(pd.DataFrame(raw).ffill().to_numpy(), nan=0.0)
    labels = decile_groups(rows, groups)
    sets = {f"D{g}": (labels == g).to_numpy() for g in range(1, groups + 1)}
    parts = []
    for name, chosen in {**sets, UNIVERSE: np.ones(len(rows), dtype=bool)}.items():
        members = np.zeros_like(opens)
        members[at[chosen], col[chosen]] = 1.0
        gross, cost, turnover, sleeves = _portfolio(members, entry, closes, horizon)
        frame = pd.DataFrame(
            {
                "date": days,
                "portfolio": name,
                "gross": gross,
                "cost": cost,
                "net": gross - cost,
                "turnover": turnover,
                "sleeves": sleeves,
            }
        )
        parts.append(frame[frame["sleeves"] > 0])
    return pd.concat(parts, ignore_index=True)[DAILY_COLUMNS]


def _monthly(net: pd.Series) -> pd.Series:
    months = pd.to_datetime(pd.Series(net.index)).dt.to_period("M").to_numpy()
    return (1 + net).groupby(months).prod() - 1


def _drawdown(excess: pd.Series) -> float:
    wealth = (1 + excess).cumprod().to_numpy()
    if not len(wealth):
        return math.nan
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float((wealth / peak - 1).min())


def _row(own: pd.DataFrame, bench: pd.DataFrame, both_legs: bool) -> dict[str, float]:
    joined = own.join(bench, how="inner", rsuffix="_bench")
    monthly, benchmark = _monthly(joined["net"]), _monthly(joined["net_bench"])
    turnover = joined["turnover"] + (joined["turnover_bench"] if both_legs else 0.0)
    return {
        "monthly_return": float(monthly.mean()),
        "monthly_excess": float((monthly - benchmark).mean()),
        "excess_drawdown": _drawdown(joined["net"] - joined["net_bench"]),
        "annual_turnover": float(turnover.mean() * ANNUAL_DAYS),
        "n_days": len(joined),
    }


def tranche_summary(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Monthly return, excess over the universe, its drawdown and turnover on days all sleeves hold.

    The last row is the top group against group 1, with both legs' turnover.
    """
    full = daily[daily["sleeves"] == horizon]
    books = {
        str(name): part.set_index("date")[["net", "turnover"]]
        for name, part in full.groupby("portfolio", sort=False)
    }
    numbered = sorted((int(name[1:]), name) for name in books if name != UNIVERSE)
    names = [name for _, name in numbered] + [UNIVERSE]
    rows = {name: _row(books[name], books[UNIVERSE], both_legs=False) for name in names}
    if len(numbered) > 1:
        low, high = numbered[0][1], numbered[-1][1]
        rows[f"{high}-{low}"] = _row(books[high], books[low], both_legs=True)
    return pd.DataFrame.from_dict(rows, orient="index")[SUMMARY_COLUMNS]
