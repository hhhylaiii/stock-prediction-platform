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
    """Group 1..groups of each row's score among its day's rows, equal counts; top scores last.

    Rows without a score get no group.
    """
    ordered = signal[signal["score"].notna()].sort_values(["date", "stock_id"])
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

    Rows must trade at the next open (untradable_t1 = 0); prices is the unfiltered price panel.
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


def _row(own: pd.Series, bench: pd.Series, turnover: pd.Series) -> dict[str, float]:
    """Monthly return, excess, excess drawdown and turnover of own against bench on shared days."""
    days = own.index.intersection(bench.index)
    monthly = _monthly(own.loc[days])
    return {
        "monthly_return": float(monthly.mean()),
        "monthly_excess": float((monthly - _monthly(bench.loc[days])).mean()),
        "excess_drawdown": _drawdown(own.loc[days] - bench.loc[days]),
        "annual_turnover": float(turnover.loc[days].mean() * ANNUAL_DAYS),
        "n_days": len(days),
    }


def tranche_summary(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Each group against the universe on steady days, then top long and group 1 short.

    Steady days hold every sleeve on the day before and after, so no full build or unwind.
    The short leg loses group 1's gross return and pays its costs; turnover adds both legs.
    """
    if not daily.empty and daily["sleeves"].max() > horizon:
        raise ValueError(f"daily holds more sleeves than horizon {horizon}")
    ordered = daily.sort_values("date", kind="stable")
    held = (ordered["sleeves"] == horizon).groupby(ordered["portfolio"])
    steady = held.shift(1, fill_value=False) & held.shift(-1, fill_value=False)
    full = ordered[(ordered["sleeves"] == horizon) & steady]
    if full.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    books = {str(name): part.set_index("date") for name, part in full.groupby("portfolio")}
    numbered = [name for _, name in sorted((int(n[1:]), n) for n in books if n != UNIVERSE)]
    universe = books[UNIVERSE]["net"]
    rows = {
        name: _row(books[name]["net"], universe, books[name]["turnover"])
        for name in [*numbered, UNIVERSE]
    }
    if len(numbered) > 1:
        high, low = books[numbered[-1]], books[numbered[0]]
        days = high.index.intersection(low.index)
        spread = high["net"].loc[days] - low["gross"].loc[days] - low["cost"].loc[days]
        legs = high["turnover"].loc[days] + low["turnover"].loc[days]
        flat = pd.Series(0.0, index=days)
        rows[f"{numbered[-1]}-{numbered[0]}"] = _row(spread, flat, legs)
    return pd.DataFrame.from_dict(rows, orient="index")[SUMMARY_COLUMNS]
