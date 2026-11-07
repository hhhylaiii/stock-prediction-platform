"""market_history rebuilt from the exchanges' stored foreign-holding pages, without fetching."""

import logging
from collections import defaultdict
from collections.abc import Iterator, Mapping
from datetime import date
from typing import NamedTuple

from django.db import transaction

from apps.etl.adapters.exchange_tables import source_rows, standard_rows
from apps.etl.errors import SourceError
from apps.etl.zero_holdings import PAGES, Page
from apps.market.market_history import market_spans
from apps.market.models import Market, MarketHistory, StockMaster

logger = logging.getLogger(__name__)
MARKETS = {"twse": Market.TWSE.value, "tpex": Market.TPEX.value}


class RebuildReport(NamedTuple):
    spans: int
    stocks: int
    moved: list[str]
    shared_days: int
    pages: dict[str, int]
    unreadable: list[tuple[str, date]]
    unsettled: list[str]


def rebuild_market_history() -> RebuildReport:
    """Codes the pages list get their spans rebuilt from the latest page of each day."""
    known = set(StockMaster.objects.values_list("stock_id", flat=True))
    listings: dict[date, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    pages = dict.fromkeys(MARKETS, 0)
    unreadable: list[tuple[str, date]] = []
    for page, day, codes in _listed_codes(unreadable):
        pages[page.source] += 1
        for code in codes & known:
            listings[day][code].add(MARKETS[page.source])
    days = [
        (day, {code: frozenset(markets) for code, markets in listed.items()})
        for day, listed in listings.items()
    ]
    spans = market_spans(days)
    counts: dict[str, int] = defaultdict(int)
    for span in spans:
        counts[span.stock_id] += 1
    with transaction.atomic():
        MarketHistory.objects.filter(stock_id__in=list(counts)).delete()
        MarketHistory.objects.bulk_create(MarketHistory(**span._asdict()) for span in spans)
    seen = {code for listed in listings.values() for code in listed}
    return RebuildReport(
        spans=len(spans),
        stocks=len(counts),
        moved=sorted(code for code, count in counts.items() if count > 1),
        shared_days=sum(len(markets) > 1 for _, listed in days for markets in listed.values()),
        pages=pages,
        unreadable=sorted(unreadable),
        unsettled=sorted(seen - set(counts)),
    )


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
