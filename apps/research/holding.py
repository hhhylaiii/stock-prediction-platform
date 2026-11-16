"""Continuous holding books, next-open trades, costs on trades (holding-shadow-prereg.md)."""

import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import pandas as pd

from apps.research.metrics.deciles import BUY_COST, SELL_COST

ENTRY = 0.90
CASH_FLOOR = 1e-12
RULE_NAME = re.compile(r"H(\d{1,2})")
DAILY_COLUMNS = ["date", "value", "cash", "paid", "traded", "held"]
ACTION_COLUMNS = ["signal_date", "date", "stock_id", "action", "amount", "limit_down_open"]
ORDER_COLUMNS = ["stock_id", "action", "priority", "score", "pct"]
BUY, SELL, SETTLE, STUCK = "buy", "sell", "settle", "stuck"
KEEP, BACKUP = "keep", "backup"
Matrix = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]


class Settle(StrEnum):
    """How an exit that cannot trade closes: live waits for delisting, known looks ahead."""

    LIVE = "live"
    KNOWN = "known"


@dataclass(frozen=True)
class Rule:
    exit_below: float
    slots: int
    entry_at: float = ENTRY

    @classmethod
    def parse(cls, name: str, slots: int) -> "Rule":
        """H70 leaves below the 0.70 percentile and enters from ENTRY."""
        matched = RULE_NAME.fullmatch(name)
        if matched is None:
            raise ValueError(f"rule must look like H70, not {name!r}")
        if slots < 1:
            raise ValueError("a book needs at least one slot")
        return cls(int(matched.group(1)) / 100, slots)

    @property
    def name(self) -> str:
        return f"H{round(self.exit_below * 100)}/N{self.slots}"


@dataclass(frozen=True)
class Costs:
    buy: float = BUY_COST
    sell: float = SELL_COST


FULL_COSTS = Costs()
NO_COSTS = Costs(0.0, 0.0)


@dataclass(frozen=True)
class Market:
    """Day by stock grids, stocks sorted by id; opens and closes are adjusted, NaN when absent.

    blocked is untradable_t1 on the signal day; delisted is the first day a stuck exit may
    settle, the day after the first signal day on or after delist_date (len(days) without one).
    """

    days: list[date]
    stocks: list[str]
    opens: Matrix
    closes: Matrix
    score: Matrix
    in_universe: Mask
    blocked: Mask
    delisted: npt.NDArray[np.int64]
    limit_down_open: Mask

    @property
    def traded(self) -> Mask:
        """A day trades with a close and an open above zero."""
        opens = np.nan_to_num(self.opens, nan=0.0)
        return np.isfinite(self.closes) & (opens > 0)

    @property
    def marks(self) -> Matrix:
        """Closes carried forward over days without one."""
        return pd.DataFrame(self.closes).ffill().to_numpy(dtype=np.float64)


@dataclass(frozen=True)
class Day:
    """One trade day of a book: the signal day's scores and that trade day's price ratios."""

    score: Matrix
    pct: Matrix
    scored: Mask
    blocked: Mask
    traded: Mask
    open_ratio: Matrix
    close_ratio: Matrix
    intraday: Matrix
    settle: Mask


@dataclass(frozen=True)
class Moves:
    sold: Mask
    settled: Mask
    stuck: Mask
    bought: Mask
    proceeds: Matrix
    entered: Matrix


@dataclass(frozen=True)
class Stepped:
    cash: float
    values: Matrix
    held: Mask
    paid: float
    traded: float
    moves: Moves


@dataclass(frozen=True)
class Book:
    daily: pd.DataFrame
    actions: pd.DataFrame
    cash: float
    values: Matrix
    held: Mask


def percentiles(score: Matrix, in_universe: Mask) -> Matrix:
    """rank(pct=True) of each row's scores among universe stocks holding one; NaN elsewhere."""
    frame = pd.DataFrame(np.where(in_universe, score, np.nan))
    return frame.rank(axis=1, pct=True).to_numpy(dtype=np.float64)


def _ratio(after: Matrix, before: Matrix, fill: float = np.nan) -> Matrix:
    usable = np.isfinite(after) & np.isfinite(before) & (before > 0)
    out = np.full(after.shape, fill)
    np.divide(after, before, out=out, where=usable)
    return out


def _idle(held: Mask) -> Moves:
    none = np.zeros_like(held)
    zero = np.zeros(held.shape)
    return Moves(none, none, none, none, zero, zero)


def _entries(kept: Mask, held: Mask, cash: float, day: Day, rule: Rule) -> tuple[Mask, float]:
    """Names not held at the open bought with the cash: best scores first, ties in stock order."""
    chosen = np.zeros_like(kept)
    free = rule.slots - int(kept.sum())
    if free <= 0 or cash <= CASH_FLOOR:
        return chosen, 0.0
    can = day.scored & (day.pct >= rule.entry_at) & ~held & ~day.blocked & day.traded
    picks = np.flatnonzero(can)
    picks = picks[np.argsort(-day.score[picks], kind="stable")][:free]
    chosen[picks] = True
    return chosen, cash / free


def step(cash: float, values: Matrix, held: Mask, day: Day, rule: Rule, costs: Costs) -> Stepped:
    """One trade day: exits at the open, entries with the cash, then the close."""
    if not day.scored.any():
        kept = np.where(held, values * day.close_ratio, 0.0)
        return Stepped(cash, kept, held, 0.0, 0.0, _idle(held))
    leave = held & ~(day.pct >= rule.exit_below)
    sold = leave & day.traded
    settled = leave & ~day.traded & day.settle
    proceeds = np.where(sold, values * day.open_ratio, np.where(settled, values, 0.0))
    gross = float(proceeds.sum())
    cash += gross * (1 - costs.sell)
    kept = held & ~(sold | settled)
    bought, each = _entries(kept, held, cash, day, rule)
    entered = np.where(bought, each / (1 + costs.buy), 0.0)
    cash -= each * int(bought.sum())
    closing = np.where(kept, values * day.close_ratio, 0.0)
    closing += np.where(bought, entered * day.intraday, 0.0)
    moves = Moves(sold, settled, leave & ~(sold | settled), bought, proceeds, entered)
    paid = gross * costs.sell + float(entered.sum()) * costs.buy
    traded = gross + float(entered.sum())
    return Stepped(cash, closing, kept | bought, paid, traded, moves)


def settle_days(market: Market, settle: Settle, traded: Mask) -> Mask:
    """Days on which each stock's stuck exit may close at its last close."""
    at = np.arange(len(market.days))[:, None]
    if settle == Settle.LIVE:
        return at >= market.delisted[None, :]
    last = np.where(traded, at, -1).max(axis=0)
    return at > last[None, :]


def _day(market: Market, pct: Matrix, marks: Matrix, traded: Mask, settles: Mask, t: int) -> Day:
    signal = t - 1
    return Day(
        score=market.score[signal],
        pct=pct[signal],
        scored=market.in_universe[signal] & np.isfinite(market.score[signal]),
        blocked=market.blocked[signal],
        traded=traded[t],
        open_ratio=_ratio(market.opens[t], marks[signal]),
        close_ratio=_ratio(marks[t], marks[signal], fill=1.0),
        intraday=_ratio(market.closes[t], market.opens[t]),
        settle=settles[t],
    )


def _actions(market: Market, t: int, moves: Moves, values: Matrix) -> list[tuple[object, ...]]:
    found = [
        (BUY, moves.bought, moves.entered),
        (SELL, moves.sold, moves.proceeds),
        (SETTLE, moves.settled, moves.proceeds),
        (STUCK, moves.stuck, values),
    ]
    return [
        (
            market.days[t - 1],
            market.days[t],
            market.stocks[k],
            action,
            float(amounts[k]),
            bool(action == SELL and market.limit_down_open[t, k]),
        )
        for action, chosen, amounts in found
        for k in np.flatnonzero(chosen)
    ]


def simulate(
    market: Market, rule: Rule, costs: Costs = FULL_COSTS, settle: Settle = Settle.LIVE
) -> Book:
    """Daily book from the first scored signal day at value 1.0, cash only, through the last day."""
    pct = percentiles(market.score, market.in_universe)
    marks, traded = market.marks, market.traded
    settles = settle_days(market, settle, traded)
    names = len(market.stocks)
    cash, values, held = 1.0, np.zeros(names), np.zeros(names, dtype=bool)
    rows: list[tuple[object, ...]] = []
    actions: list[tuple[object, ...]] = []
    started = -1
    for t in range(1, len(market.days)):
        day = _day(market, pct, marks, traded, settles, t)
        if started < 0 and day.scored.any():
            started = t
            rows.append((market.days[t - 1], 1.0, 1.0, 0.0, 0.0, 0))
        result = step(cash, values, held, day, rule, costs)
        cash, values, held = result.cash, result.values, result.held
        actions += _actions(market, t, result.moves, values)
        if started > 0:
            value = cash + float(values.sum())
            rows.append((market.days[t], value, cash, result.paid, result.traded, int(held.sum())))
    daily = pd.DataFrame(rows, columns=DAILY_COLUMNS)
    return Book(daily, pd.DataFrame(actions, columns=ACTION_COLUMNS), cash, values, held)


def order_list(
    held: Mask, score: Matrix, in_universe: Mask, stocks: list[str], rule: Rule
) -> pd.DataFrame:
    """A signal day's list before the next open: sells, keeps, then entries in priority order.

    The first free slots' worth of entries are buys and the rest backups; empty without scores.
    """
    scored = in_universe & np.isfinite(score)
    if not scored.any():
        return pd.DataFrame(columns=ORDER_COLUMNS)
    pct = percentiles(score[None, :], in_universe[None, :])[0]
    leave = held & ~(pct >= rule.exit_below)
    kept = held & ~leave
    can = scored & (pct >= rule.entry_at) & ~held
    picks = np.flatnonzero(can)
    picks = picks[np.argsort(-score[picks], kind="stable")]
    free = max(rule.slots - int(kept.sum()), 0)
    rows = [(stocks[k], SELL, 0, score[k], pct[k]) for k in np.flatnonzero(leave)]
    rows += [(stocks[k], KEEP, 0, score[k], pct[k]) for k in np.flatnonzero(kept)]
    rows += [
        (stocks[k], BUY if rank < free else BACKUP, rank + 1, score[k], pct[k])
        for rank, k in enumerate(picks)
    ]
    return pd.DataFrame(rows, columns=ORDER_COLUMNS)
