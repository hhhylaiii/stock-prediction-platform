"""Date ranges for commands that rebuild research tables piece by piece."""

from collections.abc import Iterator
from datetime import date, timedelta


def years(start: date, end: date) -> Iterator[tuple[date, date]]:
    """[start, end] cut at year boundaries."""
    for year in range(start.year, end.year + 1):
        yield max(start, date(year, 1, 1)), min(end, date(year, 12, 31))


def months(start: date, end: date) -> Iterator[tuple[date, date]]:
    """[start, end] cut at month boundaries."""
    first = date(start.year, start.month, 1)
    while first <= end:
        after = date(first.year + first.month // 12, first.month % 12 + 1, 1)
        yield max(start, first), min(end, after - timedelta(days=1))
        first = after
