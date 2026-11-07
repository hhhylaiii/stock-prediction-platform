"""Each stock's exchange on a given day, so history is not read under today's market (SPEC §8.4)."""

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from datetime import date, timedelta
from operator import itemgetter
from typing import NamedTuple

from django.db import transaction

from apps.market.models import MarketHistory

ONE_DAY = timedelta(days=1)


class Span(NamedTuple):
    stock_id: str
    start_date: date
    end_date: date | None
    market: str


def market_spans(listings: Iterable[tuple[date, Mapping[str, AbstractSet[str]]]]) -> list[Span]:
    """Spans from the markets listing each code each day; a day listed on both keeps the old one.

    The exchanges list a stock the day before it first trades there, so the day both list it is
    its last on the old market.
    """
    opened: dict[str, tuple[date, str]] = {}
    unsettled: dict[str, date] = {}
    closed: list[Span] = []
    for day, listed in sorted(listings, key=itemgetter(0)):
        for code, markets in listed.items():
            current = opened.get(code)
            if len(markets) != 1:
                if current is None:
                    unsettled.setdefault(code, day)
                continue
            (market,) = markets
            if current is None:
                opened[code] = (unsettled.pop(code, day), market)
            elif current[1] != market:
                closed.append(Span(code, current[0], day - ONE_DAY, current[1]))
                opened[code] = (day, market)
    spans = [
        *closed,
        *(Span(code, start, None, market) for code, (start, market) in opened.items()),
    ]
    return sorted(spans, key=itemgetter(0, 1))


def merge_spans(
    pages: Sequence[Span], stored: Sequence[Span], seen: AbstractSet[date]
) -> list[Span]:
    """One stock's page spans, plus the stored boundaries on days the pages could not see."""
    starts = [(span.start_date, span.market) for span in pages]
    starts += [(span.start_date, span.market) for span in stored if span.start_date not in seen]
    merged: list[tuple[date, str]] = []
    for start, market in sorted(starts):
        if not merged or merged[-1][1] != market:
            merged.append((start, market))
    code = (pages or stored)[0].stock_id
    ends = [start - ONE_DAY for start, _ in merged[1:]]
    return [
        Span(code, start, end, market)
        for (start, market), end in zip(merged, [*ends, None], strict=True)
    ]


def record_markets(day: date, markets: Mapping[str, str]) -> list[str]:
    """New codes, and codes quoted off their open span's market, get a span from day."""
    current = {
        stock_id: (start, market)
        for stock_id, start, market in MarketHistory.objects.filter(
            stock_id__in=list(markets), end_date__isnull=True
        ).values_list("stock_id", "start_date", "market")
    }
    moved = sorted(
        code
        for code, market in markets.items()
        if code in current and current[code][1] != market and current[code][0] < day
    )
    new = sorted(code for code in markets if code not in current)
    with transaction.atomic():
        MarketHistory.objects.filter(stock_id__in=moved, end_date__isnull=True).update(
            end_date=day - ONE_DAY
        )
        MarketHistory.objects.bulk_create(
            MarketHistory(stock_id=code, start_date=day, market=markets[code])
            for code in [*moved, *new]
        )
    return sorted([*moved, *new])
