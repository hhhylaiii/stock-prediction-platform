"""First day a stock traded on TWSE or TPEx, so emerging-market rows stay out (SPEC §1, §8.6)."""

from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from typing import NamedTuple, Protocol

MONTHS = 12


class Outcome(StrEnum):
    LISTED = "listed"
    EARLIER = "listed_before_prices"
    UNKNOWN = "unanswerable"
    MISSING = "no official trading day up to the last price"
    FIRST_PRICE = "first_price_after_empty_month"


class Listing(NamedTuple):
    outcome: Outcome
    day: date | None = None


class Probe(Protocol):
    def __call__(self, month: date, every: bool = False) -> list[date] | None: ...


class Ask(Protocol):
    def __call__(self, month: date, every: bool = False) -> list[date]: ...


class _UnanswerableError(Exception):
    pass


def first_listed_day(firsts: Sequence[date], probe: Probe) -> Listing:
    """Search only months with prices; firsts holds each such month's first price day, ascending."""

    def ask(month: date, every: bool = False) -> list[date]:
        days = probe(month, every)
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
    """Prices before the month's first day on either market are emerging trades."""
    if days[0] > first_price:
        days = ask(_month(first_price), every=True)
        if days[0] > first_price:
            return Listing(Outcome.LISTED, days[0])
    if ask(_shift(_month(first_price), -1)):
        return Listing(Outcome.EARLIER)
    return Listing(Outcome.FIRST_PRICE, days[0])


def _later_month(firsts: Sequence[date], ask: Ask) -> Listing:
    """Gallop then bisect over the price months, after the first one came back empty."""
    empty, step, last = 0, 1, len(firsts) - 1
    while empty < last:
        index = min(empty + step, last)
        if ask(_month(firsts[index])):
            break
        empty, step = index, step * 2
    else:
        return Listing(Outcome.MISSING)
    listed = index
    while listed - empty > 1:
        middle = (empty + listed) // 2
        if ask(_month(firsts[middle])):
            listed = middle
        else:
            empty = middle
    return Listing(Outcome.LISTED, ask(_month(firsts[listed]), every=True)[0])


def _month(day: date) -> date:
    return day.replace(day=1)


def _shift(month: date, months: int) -> date:
    index = month.year * MONTHS + month.month - 1 + months
    return date(index // MONTHS, index % MONTHS + 1, 1)
