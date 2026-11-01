"""Set stock_master.list_date from official monthly trading days; drop rows outside listing."""

from argparse import ArgumentParser
from collections.abc import Callable
from datetime import date
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import Min, Q, QuerySet
from django.db.models.functions import TruncMonth
from django.utils import timezone

from apps.etl.adapters import limiter_for
from apps.etl.adapters.tpex import TpexClient
from apps.etl.adapters.twse import TwseClient
from apps.etl.errors import SourceError
from apps.etl.listing import Outcome, Probe, first_listed_day
from apps.etl.models import EtlRun
from apps.etl.ratelimit import RetryableError
from apps.etl.runs import failing_on_termination, tracked_run
from apps.market.models import DailyChip, DailyPrice, DailyValuation, Market, StockMaster

DATASET = "list_date"
MARKET_ROWS = (DailyPrice, DailyValuation, DailyChip)
Lookup = Callable[[str, date], list[date] | None]


class Command(BaseCommand):
    help = (
        "Find each stock's first TWSE or TPEx trading day and drop market rows before it or "
        "from delist_date on; run after the FinMind daily_price backfill."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--stock", action="append", dest="stocks")
        parser.add_argument("--refresh", action="store_true")

    def handle(self, *args: Any, **options: Any) -> None:
        chosen = StockMaster.objects.order_by("stock_id")
        if options["stocks"]:
            chosen = chosen.filter(stock_id__in=options["stocks"])
        self._drop_after_delisting(chosen)
        refresh = options["refresh"]
        stocks = chosen if refresh else chosen.filter(list_date__isnull=True)
        firsts = _price_months(stocks)
        settled = {} if refresh else _settled()
        todo = [
            stock
            for stock in stocks
            if stock.stock_id in firsts
            and settled.get(stock.stock_id) != firsts[stock.stock_id][0].isoformat()
        ]
        today = timezone.localdate()
        start = min((days[0] for days in firsts.values()), default=today)
        clients = (TwseClient.build(limiter_for("twse")), TpexClient.build(limiter_for("tpex")))
        try:
            with failing_on_termination(), tracked_run("twse_tpex", DATASET, start, today) as run:
                run.checkpoint = {"rejected": {}, "settled": {}, "cleared": []}
                for stock in todo:
                    self._settle(run, stock, firsts[stock.stock_id], _probe(stock, *clients))
                _fail_on_rejections(run.checkpoint["rejected"])
        except RetryableError as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc

    def _settle(self, run: EtlRun, stock: StockMaster, firsts: list[date], probe: Probe) -> None:
        code, progress = stock.stock_id, run.checkpoint
        try:
            listing = first_listed_day(firsts, probe)
        except SourceError as exc:
            progress["rejected"][code] = f"{type(exc).__name__}: {exc}"
        else:
            if listing.outcome is Outcome.LISTED and listing.day is not None:
                self._set(stock, listing.day)
                run.rows += 1
            elif listing.outcome is Outcome.MISSING:
                progress["rejected"][code] = str(listing.outcome)
            else:
                progress["settled"][code] = [firsts[0].isoformat(), str(listing.outcome)]
                if stock.list_date is not None:
                    self._clear(stock)
                    progress["cleared"].append(code)
        run.save(update_fields=["checkpoint", "rows"])

    def _set(self, stock: StockMaster, listed: date) -> None:
        with transaction.atomic():
            stock.list_date = listed
            stock.save(update_fields=["list_date"])
            dropped = _drop(Q(stock_id=stock.stock_id, date__lt=listed))
        self.stdout.write(f"{stock.stock_id}: listed {listed}, dropped {dropped} earlier rows")

    def _clear(self, stock: StockMaster) -> None:
        old, stock.list_date = stock.list_date, None
        stock.save(update_fields=["list_date"])
        self.stdout.write(
            f"{stock.stock_id}: listing date {old} no longer confirmed and cleared; rerun the "
            "FinMind backfill of daily_price, daily_valuation and daily_chip for it"
        )

    def _drop_after_delisting(self, stocks: QuerySet[StockMaster]) -> None:
        for stock in stocks.filter(delist_date__isnull=False):
            dropped = _drop(Q(stock_id=stock.stock_id, date__gte=stock.delist_date))
            if dropped:
                self.stdout.write(f"{stock.stock_id}: dropped {dropped} rows from delisting on")


def _drop(rows: Q) -> int:
    return sum(model.objects.filter(rows).delete()[0] for model in MARKET_ROWS)


def _price_months(stocks: QuerySet[StockMaster]) -> dict[str, list[date]]:
    """First traded day of every month with prices, per stock."""
    rows = (
        DailyPrice.objects.filter(stock_id__in=stocks.values("stock_id"), volume__gt=0)
        .annotate(month=TruncMonth("date"))
        .values("stock_id", "month")
        .annotate(first=Min("date"))
        .order_by("stock_id", "month")
    )
    firsts: dict[str, list[date]] = {}
    for row in rows:
        firsts.setdefault(row["stock_id"], []).append(row["first"])
    return firsts


def _settled() -> dict[str, str]:
    """Stocks an earlier run found listed before, or unanswerable from, the same first price."""
    found: dict[str, str] = {}
    for checkpoint in (
        EtlRun.objects.filter(dataset=DATASET)
        .order_by("run_id")
        .values_list("checkpoint", flat=True)
    ):
        found |= {code: entry[0] for code, entry in checkpoint.get("settled", {}).items()}
    return found


def _fail_on_rejections(rejected: dict[str, str]) -> None:
    if rejected:
        noun = "stock" if len(rejected) == 1 else "stocks"
        raise CommandError(f"{len(rejected)} {noun} rejected: {', '.join(sorted(rejected))}")


def _probe(stock: StockMaster, twse: TwseClient, tpex: TpexClient) -> Probe:
    """Ask the current market first, then the other; None when no market could answer."""
    lookups: list[Lookup] = [twse.stock_month, tpex.stock_month]
    if stock.market == Market.TPEX:
        lookups.reverse()

    def probe(month: date) -> list[date] | None:
        answers = []
        for lookup in lookups:
            days = lookup(stock.stock_id, month)
            if days:
                return days
            answers.append(days)
        return None if None in answers else []

    return probe
