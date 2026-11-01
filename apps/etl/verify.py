"""Compare stored daily prices with the official per-stock monthly tables (SPEC §9 Phase 1)."""

import random
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import NamedTuple

from django.db.models.functions import ExtractYear

from apps.etl.exchange_rows import OfficialDay
from apps.market.models import DailyPrice

FIRST_YEAR = 2010
Stored = tuple[Decimal | None, int]


class Difference(NamedTuple):
    stock_id: str
    day: date
    field: str
    stored: str
    official: str


def sample_pairs(count: int, years: int, seed: int, last_year: int) -> list[tuple[str, int]]:
    """count stocks with at least years priced years, then years of each, from a fixed seed."""
    priced = (
        DailyPrice.objects.filter(date__year__gte=FIRST_YEAR, date__year__lte=last_year)
        .annotate(year=ExtractYear("date"))
        .values_list("stock_id", "year")
        .distinct()
    )
    years_of: dict[str, list[int]] = {}
    for stock_id, year in priced:
        years_of.setdefault(stock_id, []).append(year)
    years_of = {code: found for code, found in years_of.items() if len(found) >= years}
    rng = random.Random(seed)  # noqa: S311
    stocks = sorted(rng.sample(sorted(years_of), min(count, len(years_of))))
    pairs = []
    for stock_id in stocks:
        priced_years = sorted(years_of[stock_id])
        chosen = rng.sample(priced_years, years)
        pairs += [(stock_id, year) for year in sorted(chosen)]
    return pairs


def stored_days(stock_id: str, first: date, last: date) -> dict[date, Stored]:
    rows = DailyPrice.objects.filter(stock_id=stock_id, date__range=(first, last))
    return {
        day: (close, volume) for day, close, volume in rows.values_list("date", "close", "volume")
    }


def compare_days(
    stock_id: str, official: Mapping[date, OfficialDay], stored: Mapping[date, Stored]
) -> list[Difference]:
    """Close must match exactly; volume within half the official unit (TPEx rounds to 1,000)."""
    differences = []
    for day in sorted(official.keys() | stored.keys()):
        if day not in stored:
            mark = f"{official[day].close} / {official[day].volume}"
            differences.append(Difference(stock_id, day, "missing", "-", mark))
        elif day not in official:
            close, volume = stored[day]
            differences.append(Difference(stock_id, day, "extra", f"{close} / {volume}", "-"))
        else:
            differences.extend(_field_differences(stock_id, official[day], stored[day]))
    return differences


def _field_differences(stock_id: str, official: OfficialDay, stored: Stored) -> list[Difference]:
    close, volume = stored
    found = []
    if close != official.close:
        found.append(Difference(stock_id, official.day, "close", str(close), str(official.close)))
    if abs(volume - official.volume) * 2 > official.volume_unit:
        found.append(
            Difference(stock_id, official.day, "volume", str(volume), str(official.volume))
        )
    return found
