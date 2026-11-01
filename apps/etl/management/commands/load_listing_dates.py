"""Set stock_master.list_date from official monthly trading days and drop earlier rows.

Stocks already trading the month before their first price stay without a date: their listing
predates the price history, so no emerging-market rows can be in it.
"""

from argparse import ArgumentParser
from datetime import date
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import Max, Min
from django.utils import timezone

from apps.etl.adapters import limiter_for
from apps.etl.adapters.tpex import TpexClient
from apps.etl.adapters.twse import STOCK_DAY_FIRST_MONTH, TwseClient
from apps.etl.errors import SourceError
from apps.etl.listing import Probe, first_listed_day
from apps.etl.runs import tracked_run
from apps.market.models import DailyChip, DailyPrice, DailyValuation, Market, StockMaster

MARKET_ROWS = (DailyPrice, DailyValuation, DailyChip)


class Command(BaseCommand):
    help = "Find each stock's first TWSE or TPEx trading day and drop rows dated before it."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--stock", action="append", dest="stocks")
        parser.add_argument("--refresh", action="store_true")

    def handle(self, *args: Any, **options: Any) -> None:
        stocks = StockMaster.objects.all()
        if options["stocks"]:
            stocks = stocks.filter(stock_id__in=options["stocks"])
        if not options["refresh"]:
            stocks = stocks.filter(list_date__isnull=True)
        spans = {
            row["stock_id"]: (row["first"], row["last"])
            for row in DailyPrice.objects.filter(stock_id__in=stocks.values("stock_id"))
            .values("stock_id")
            .annotate(first=Min("date"), last=Max("date"))
        }
        twse = TwseClient.build(limiter_for("twse"))
        tpex = TpexClient.build(limiter_for("tpex"))
        today = timezone.localdate()
        first = min((span[0] for span in spans.values()), default=today)
        rejected: dict[str, str] = {}
        earlier: list[str] = []
        with tracked_run("twse_tpex", "list_date", first, today) as run:
            for stock in stocks.filter(stock_id__in=spans).order_by("stock_id"):
                try:
                    listing = first_listed_day(*spans[stock.stock_id], _probe(stock, twse, tpex))
                except SourceError as exc:
                    rejected[stock.stock_id] = f"{type(exc).__name__}: {exc}"
                    continue
                if listing.before_hint:
                    earlier.append(stock.stock_id)
                elif listing.day is None:
                    rejected[stock.stock_id] = "no official trading day up to the last price"
                else:
                    self._set(stock, listing.day)
                    run.rows += 1
            run.checkpoint = {"rejected": rejected, "listed_before_prices": earlier}
            if rejected:
                codes = ", ".join(sorted(rejected))
                noun = "stock" if len(rejected) == 1 else "stocks"
                raise CommandError(f"{len(rejected)} {noun} rejected: {codes}")

    def _set(self, stock: StockMaster, listed: date) -> None:
        with transaction.atomic():
            stock.list_date = listed
            stock.save(update_fields=["list_date"])
            dropped = sum(
                model.objects.filter(stock_id=stock.stock_id, date__lt=listed).delete()[0]
                for model in MARKET_ROWS
            )
        self.stdout.write(f"{stock.stock_id}: listed {listed}, dropped {dropped} earlier rows")


def _probe(stock: StockMaster, twse: TwseClient, tpex: TpexClient) -> Probe:
    """A TWSE stock may have traded on TPEx first; TWSE cannot answer before 2010, so None."""
    on_twse = stock.market == Market.TWSE

    def probe(month: date) -> list[date] | None:
        if on_twse and month >= STOCK_DAY_FIRST_MONTH:
            days = twse.stock_month(stock.stock_id, month)
            if days:
                return days
        days = tpex.stock_month(stock.stock_id, month)
        if days or not on_twse or month >= STOCK_DAY_FIRST_MONTH:
            return days
        return None

    return probe
