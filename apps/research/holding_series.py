"""A shadow book's cumulative value beside the equal-weight universe and TAIEX_TR."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Fill:
    """One fill row: the book's value after the trade day's close and that day's two returns."""

    trade_date: date
    value: float
    universe_return: float | None
    index_return: float | None


@dataclass(frozen=True)
class Point:
    """Cumulative values on a trade day, all 1.0 on the first; None once a return went missing."""

    trade_date: date
    book: float
    universe: float | None
    index: float | None


def cumulative(fills: Iterable[Fill]) -> list[Point]:
    """The book as value ÷ first value; the universe and index chained from the next day on.

    The first day's two returns are close to close while the book bought at that open, so they are
    left out, and a missing return blanks that series from its day on instead of counting as 0.
    """
    ordered = sorted(fills, key=lambda fill: fill.trade_date)
    if not ordered:
        return []
    _check(ordered)
    first = ordered[0]
    points = [Point(first.trade_date, 1.0, 1.0, 1.0)]
    for fill in ordered[1:]:
        last = points[-1]
        points.append(
            Point(
                fill.trade_date,
                fill.value / first.value,
                _chain(last.universe, fill.universe_return),
                _chain(last.index, fill.index_return),
            )
        )
    return points


def _chain(level: float | None, change: float | None) -> float | None:
    return None if level is None or change is None else level * (1 + change)


def _check(ordered: list[Fill]) -> None:
    days = [fill.trade_date for fill in ordered]
    if len(set(days)) != len(days):
        raise ValueError("two fills on the same trade day")
    for fill in ordered:
        if not math.isfinite(fill.value) or fill.value <= 0:
            raise ValueError(f"fill {fill.trade_date} has value {fill.value}, not positive")
        for change in (fill.universe_return, fill.index_return):
            if change is not None and (not math.isfinite(change) or change <= -1):
                raise ValueError(f"fill {fill.trade_date} has return {change}")
