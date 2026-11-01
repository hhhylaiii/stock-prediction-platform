"""First day a stock traded on TWSE or TPEx, so emerging-market rows stay out (SPEC §1, §8.6)."""

from collections.abc import Callable
from datetime import date
from typing import NamedTuple

MONTHS = 12
Probe = Callable[[date], list[date] | None]


class Listing(NamedTuple):
    day: date | None
    before_hint: bool = False


def first_listed_day(hint: date, last: date, probe: Probe) -> Listing:
    """First official trading day after an empty month; an unanswerable month counts as traded."""
    start, end = _month(hint), _month(last)
    found = probe(start)
    if found:
        earlier = probe(_shift(start, -1))
        return Listing(found[0]) if earlier == [] else Listing(None, True)
    if start >= end:
        return Listing(None)
    empty, step = start, 1
    while True:
        candidate = min(_shift(start, step), end)
        found = probe(candidate)
        if found:
            break
        if candidate == end:
            return Listing(None)
        empty, step = candidate, step * 2
    listed = candidate
    while _gap(empty, listed) > 1:
        middle = _shift(empty, _gap(empty, listed) // 2)
        days = probe(middle)
        if days:
            listed, found = middle, days
        else:
            empty = middle
    return Listing(found[0])


def _month(day: date) -> date:
    return day.replace(day=1)


def _shift(month: date, months: int) -> date:
    index = month.year * MONTHS + month.month - 1 + months
    return date(index // MONTHS, index % MONTHS + 1, 1)


def _gap(earlier: date, later: date) -> int:
    return (later.year - earlier.year) * MONTHS + later.month - earlier.month
