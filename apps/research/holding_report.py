"""Holding book summaries against the costless equal-weight universe and TAIEX_TR."""

from datetime import date

import numpy as np
import pandas as pd

from apps.research.holding import Market, Mask, Matrix
from apps.research.metrics.stats import newey_west_t

MONTH_DAYS = 21
YEAR_DAYS = 252
NW_LAG = 5
SUMMARY_COLUMNS = [
    "net_mo",
    "cost_mo",
    "ew_mo",
    "excess_mo",
    "excess_t",
    "index_mo",
    "net_cagr",
    "ew_cagr",
    "index_cagr",
    "drawdown",
    "turnover_yr",
    "held",
    "days",
]


def universe_returns(market: Market) -> Matrix:
    """Mean close-to-close return of the day before's universe; carried closes, 0 when empty."""
    out = np.zeros(len(market.days))
    out[1:] = mean_returns(market.marks, market.in_universe)
    return out


def mean_returns(marks: Matrix, in_universe: Mask) -> Matrix:
    """Each next row's mean return over the row's universe stocks marked on both; 0 for none."""
    before, after = marks[:-1], marks[1:]
    counted = in_universe[:-1] & np.isfinite(before) & np.isfinite(after)
    change = np.zeros(after.shape)
    np.divide(after, before, out=change, where=counted & (before > 0))
    sums = np.where(counted, change - 1, 0.0).sum(axis=1)
    counts = counted.sum(axis=1)
    means: Matrix = np.divide(sums, counts, out=np.zeros(len(sums)), where=counts > 0)
    return means


def index_returns(levels: pd.Series, days: list[date]) -> Matrix:
    """Close-to-close returns of an index level on the days, carried over missing closes."""
    level = levels.reindex(days).ffill()
    return np.asarray(level.pct_change().fillna(0.0).to_numpy(), dtype=np.float64)


def _compound(returns: Matrix, years: float) -> float:
    return float(float(np.prod(1 + returns)) ** (1 / years) * 100 - 100)


def summarise(
    daily: pd.DataFrame, days: list[date], universe: Matrix, index: Matrix
) -> dict[str, float]:
    """Monthly means in %, NW t of the excess, annualised growth, drawdown and turnover."""
    at = pd.Index(days).get_indexer(pd.Index(daily["date"]))[1:]
    value = daily["value"].to_numpy(dtype=np.float64)
    net = value[1:] / value[:-1] - 1
    cost = daily["paid"].to_numpy(dtype=np.float64)[1:] / value[:-1]
    turnover = daily["traded"].to_numpy(dtype=np.float64)[1:] / value[:-1] / 2
    excess = net - universe[at]
    years = len(net) / YEAR_DAYS
    peak = np.maximum.accumulate(value)
    return {
        "net_mo": net.mean() * MONTH_DAYS * 100,
        "cost_mo": cost.mean() * MONTH_DAYS * 100,
        "ew_mo": universe[at].mean() * MONTH_DAYS * 100,
        "excess_mo": excess.mean() * MONTH_DAYS * 100,
        "excess_t": newey_west_t(excess, NW_LAG),
        "index_mo": index[at].mean() * MONTH_DAYS * 100,
        "net_cagr": (value[-1] / value[0]) ** (1 / years) * 100 - 100,
        "ew_cagr": _compound(universe[at], years),
        "index_cagr": _compound(index[at], years),
        "drawdown": float((value / peak - 1).min()) * 100,
        "turnover_yr": turnover.mean() * YEAR_DAYS,
        "held": daily["held"].to_numpy(dtype=np.float64)[1:].mean(),
        "days": float(len(net)),
    }


def by_year(daily: pd.DataFrame, days: list[date], universe: Matrix, index: Matrix) -> pd.DataFrame:
    """summarise per calendar year, each year starting from the previous year's last value."""
    years = pd.to_datetime(daily["date"]).dt.year
    rows = {}
    for year in sorted(set(years)):
        part = daily[(years == year) | (years.shift(-1) == year)]
        if len(part) > 2:
            rows[str(year)] = summarise(part.reset_index(drop=True), days, universe, index)
    return pd.DataFrame.from_dict(rows, orient="index", columns=SUMMARY_COLUMNS)
