"""A shadow book's cumulative value beside the equal-weight universe and TAIEX_TR."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from itertools import pairwise


@dataclass(frozen=True)
class Fill:
    """One fill row with the TAIEX_TR close of its trade day, read from `index_daily`."""

    signal_date: date
    trade_date: date
    value: float
    universe_return: float | None
    index_close: float | None


@dataclass(frozen=True)
class Point:
    """Cumulative values on a trade day; None where a figure is missing."""

    trade_date: date
    book: float
    universe: float | None
    index: float | None


def cumulative(fills: Iterable[Fill]) -> list[Point]:
    """The book and index as ratios to the first day; the universe chained from the next day on.

    A missing universe return blanks the universe from that day on, a missing close only its day,
    or the whole index when it is the first day's.
    """
    ordered = sorted(fills, key=lambda fill: fill.trade_date)
    if not ordered:
        return []
    _check(ordered)
    first = ordered[0]
    points = [Point(first.trade_date, 1.0, 1.0, _ratio(first.index_close, first.index_close))]
    for fill in ordered[1:]:
        last = points[-1]
        universe = _chain(last.universe, fill.universe_return)
        index = _ratio(fill.index_close, first.index_close)
        points.append(Point(fill.trade_date, fill.value / first.value, universe, index))
    return points


def _chain(level: float | None, change: float | None) -> float | None:
    return None if level is None or change is None else level * (1 + change)


def _ratio(close: float | None, base: float | None) -> float | None:
    return None if close is None or base is None else close / base


def _check(ordered: list[Fill]) -> None:
    days = [fill.trade_date for fill in ordered]
    if len(set(days)) != len(days):
        raise ValueError("two fills on the same trade day")
    for before, fill in pairwise(ordered):
        if fill.signal_date != before.trade_date:
            raise ValueError(f"gap before {fill.trade_date}: it signed on {fill.signal_date}")
    for fill in ordered:
        if not math.isfinite(fill.value) or fill.value <= 0:
            raise ValueError(f"fill {fill.trade_date} has value {fill.value}, not positive")
        if fill.universe_return is not None and not -1 < fill.universe_return < math.inf:
            raise ValueError(f"fill {fill.trade_date} has universe return {fill.universe_return}")
        if fill.index_close is not None and not 0 < fill.index_close < math.inf:
            raise ValueError(f"fill {fill.trade_date} has index close {fill.index_close}")
