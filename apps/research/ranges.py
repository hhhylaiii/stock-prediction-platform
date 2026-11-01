"""Date ranges for commands that rebuild research tables piece by piece."""

from collections.abc import Iterator
from datetime import date


def years(start: date, end: date) -> Iterator[tuple[date, date]]:
    """[start, end] cut at year boundaries."""
    for year in range(start.year, end.year + 1):
        yield max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
