"""market_history rebuilt from the exchanges' stored foreign-holding pages, without fetching."""

import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from datetime import date
from typing import Any, NamedTuple

from django.db import transaction
from django.utils import timezone

from apps.etl.adapters.exchange_tables import source_rows, standard_rows
from apps.etl.errors import SourceError
from apps.etl.models import DataQualityLog
from apps.etl.zero_holdings import PAGES, Page
from apps.market.calendar import TradingDays
from apps.market.market_history import Span, market_spans
from apps.market.models import Market, MarketHistory, StockMaster

logger = logging.getLogger(__name__)
MARKETS = {"twse": Market.TWSE.value, "tpex": Market.TPEX.value}
ONLY = {source: frozenset({market}) for source, market in MARKETS.items()}
NONE: frozenset[str] = frozenset()
REBUILT_CHECK = "market_history_rebuilt"
SAMPLE_SIZE = 20
Listings = dict[date, dict[str, set[str]]]
Spans = list[tuple[date, date | None, str]]


class RebuildReport(NamedTuple):
    spans: int
    stocks: int
    moved: list[str]
    shared_days: int
    pages: dict[str, int]
    unreadable: list[tuple[str, date]]
    missing: dict[str, list[date]]
    unsettled: list[str]
    changed: list[str]


def rebuild_market_history(calendar: TradingDays) -> RebuildReport:
    """Codes the pages list get their spans rebuilt from the days both exchanges' pages are stored.

    Spans that differ from the stored ones are logged, as the daily quotes may have written them.
    """
    listings, unreadable = _listings()
    spans = market_spans((day, _markets(listed)) for day, listed in _both(listings))
    rebuilt: dict[str, Spans] = defaultdict(list)
    for span in spans:
        rebuilt[span.stock_id].append((span.start_date, span.end_date, span.market))
    with transaction.atomic():
        changed = _replace(rebuilt, spans)
    seen = {code for listed in listings.values() for codes in listed.values() for code in codes}
    days = {
        source: {day for day, listed in listings.items() if source in listed} for source in MARKETS
    }
    return RebuildReport(
        spans=len(spans),
        stocks=len(rebuilt),
        moved=sorted(code for code, rows in rebuilt.items() if len(rows) > 1),
        shared_days=sum(len(set.intersection(*listed.values())) for _, listed in _both(listings)),
        pages={source: len(on) for source, on in days.items()},
        unreadable=sorted(unreadable),
        missing=_missing(days, calendar),
        unsettled=sorted(seen - set(rebuilt)),
        changed=changed,
    )


def _listings() -> tuple[Listings, list[tuple[str, date]]]:
    """The stock_master codes on each day's page, by exchange."""
    known = set(StockMaster.objects.values_list("stock_id", flat=True))
    listings: Listings = defaultdict(dict)
    unreadable: list[tuple[str, date]] = []
    for page, day, codes in _listed_codes(unreadable):
        listings[day][page.source] = codes & known
    return listings, unreadable


def _markets(listed: Mapping[str, set[str]]) -> dict[str, frozenset[str]]:
    markets: dict[str, frozenset[str]] = {}
    for source, codes in listed.items():
        for code in codes:
            markets[code] = markets.get(code, NONE) | ONLY[source]
    return markets


def _both(listings: Listings) -> Iterator[tuple[date, dict[str, set[str]]]]:
    """Days with both exchanges' pages, in order; a one-sided day cannot tell a move's last day."""
    return ((day, listings[day]) for day in sorted(listings) if len(listings[day]) == len(MARKETS))


def _replace(rebuilt: Mapping[str, Spans], spans: Iterable[Span]) -> list[str]:
    """Swap in the rebuilt spans and log the codes whose stored spans differed."""
    stored: dict[str, Spans] = defaultdict(list)
    rows = MarketHistory.objects.filter(stock_id__in=list(rebuilt)).order_by(
        "stock_id", "start_date"
    )
    for stock_id, start, end, market in rows.values_list(
        "stock_id", "start_date", "end_date", "market"
    ):
        stored[stock_id].append((start, end, market))
    changed = sorted(code for code, old in stored.items() if old != rebuilt[code])
    rows.delete()
    MarketHistory.objects.bulk_create(MarketHistory(**span._asdict()) for span in spans)
    if changed:
        sample = [[code, _plain(stored[code]), _plain(rebuilt[code])] for code in changed]
        DataQualityLog.objects.create(
            date=timezone.localdate(),
            check_name=REBUILT_CHECK,
            severity=DataQualityLog.Severity.INFO,
            detail={"changed": len(changed), "sample": sample[:SAMPLE_SIZE]},
        )
    return changed


def _plain(spans: Spans) -> list[list[Any]]:
    return [
        [start.isoformat(), end.isoformat() if end else None, market]
        for start, end, market in spans
    ]


def _missing(seen_days: Mapping[str, set[date]], calendar: TradingDays) -> dict[str, list[date]]:
    """Trading days between the first and last page of either exchange that one of them lacks."""
    every = set().union(*seen_days.values())
    if not every:
        return {source: [] for source in seen_days}
    sessions = calendar.between(min(every), max(every))
    return {
        source: [day for day in sessions if day not in days] for source, days in seen_days.items()
    }


def _listed_codes(unreadable: list[tuple[str, date]]) -> Iterator[tuple[Page, date, set[str]]]:
    """Codes on the latest stored page of each day and exchange; unreadable pages are skipped."""
    for page in PAGES:
        for day, raw_id in sorted(_latest_pages(page).items()):
            payload = page.raw_model._default_manager.values_list("payload", flat=True).get(
                pk=raw_id
            )
            try:
                rows = source_rows(payload, day, page.table, page.check)
                frame = standard_rows(rows, page.table._replace(valid={}))
            except SourceError as exc:
                logger.warning("%s raw %s for %s is unreadable: %s", page.source, raw_id, day, exc)
                unreadable.append((page.source, day))
                continue
            yield page, day, set(frame["stock_id"])


def _latest_pages(page: Page) -> Mapping[date, int]:
    rows = page.raw_model._default_manager.filter(dataset="foreign_holding").order_by("fetched_at")
    latest: dict[date, int] = {}
    for params, raw_id in rows.values_list("request_params", "pk"):
        latest[date.fromisoformat(params["date"].replace("/", "-"))] = raw_id
    return latest
