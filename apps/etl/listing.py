"""First day a stock traded on TWSE or TPEx, so emerging-market rows stay out (SPEC §1, §8.6)."""

from collections.abc import Callable, Sequence
from datetime import date
from enum import StrEnum
from typing import NamedTuple

MONTHS = 12
Probe = Callable[[date], list[date] | None]
Ask = Callable[[date], list[date]]


class Outcome(StrEnum):
    LISTED = "listed"
    EARLIER = "listed_before_prices"
    UNKNOWN = "unanswerable"
    MISSING = "no official trading day up to the last price"


class Listing(NamedTuple):
    outcome: Outcome
    day: date | None = None


class _UnanswerableError(Exception):
    pass


def first_listed_day(firsts: Sequence[date], probe: Probe) -> Listing:
    """Search only months with prices; firsts holds each such month's first price day, ascending."""

    def ask(month: date) -> list[date]:
        days = probe(month)
        if days is None:
            raise _UnanswerableError
        return days

    try:
        days = ask(_month(firsts[0]))
        if days:
            return _first_month(firsts[0], days, ask)
        return _later_month(firsts, ask)
    except _UnanswerableError:
        return Listing(Outcome.UNKNOWN)


def _first_month(first_price: date, days: list[date], ask: Ask) -> Listing:
    """Prices before the first official day of their month are emerging trades."""
    if days[0] > first_price:
        return Listing(Outcome.LISTED, days[0])
    if ask(_shift(_month(first_price), -1)):
        return Listing(Outcome.EARLIER)
    return Listing(Outcome.LISTED, days[0])


def _later_month(firsts: Sequence[date], ask: Ask) -> Listing:
    """Gallop then bisect over the price months, after the first one came back empty."""
    empty, step, last = 0, 1, len(firsts) - 1
    while empty < last:
        index = min(empty + step, last)
        found = ask(_month(firsts[index]))
        if found:
            break
        empty, step = index, step * 2
    else:
        return Listing(Outcome.MISSING)
    listed = index
    while listed - empty > 1:
        middle = (empty + listed) // 2
        days = ask(_month(firsts[middle]))
        if days:
            listed, found = middle, days
        else:
            empty = middle
    return Listing(Outcome.LISTED, found[0])


def _month(day: date) -> date:
    return day.replace(day=1)


def _shift(month: date, months: int) -> date:
    index = month.year * MONTHS + month.month - 1 + months
    return date(index // MONTHS, index % MONTHS + 1, 1)
