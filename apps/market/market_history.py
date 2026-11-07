"""Each stock's exchange on a given day, so history is not read under today's market (SPEC §8.4)."""

from bisect import bisect_right
from collections import defaultdict
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


def market_on(stock_ids: Sequence[str], days: Sequence[date]) -> list[str | None]:
    """The market of each stock on the day paired with it; None where no span covers the day."""
    spans: dict[str, list[tuple[date, date | None, str]]] = defaultdict(list)
    rows = MarketHistory.objects.filter(stock_id__in=set(stock_ids)).order_by(
        "stock_id", "start_date"
    )
    for stock_id, start, end, market in rows.values_list(
        "stock_id", "start_date", "end_date", "market"
    ):
        spans[stock_id].append((start, end, market))
    return [_covering(spans.get(code, []), day) for code, day in zip(stock_ids, days, strict=True)]


def _covering(spans: list[tuple[date, date | None, str]], day: date) -> str | None:
    index = bisect_right([start for start, _, _ in spans], day) - 1
    if index < 0:
        return None
    _, end, market = spans[index]
    return market if end is None or day <= end else None


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
