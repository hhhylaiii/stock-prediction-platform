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
from apps.etl.errors import RejectedStocksError, SourceError
from apps.etl.listing import Outcome, Probe, first_listed_day
from apps.etl.models import EtlRun
from apps.etl.ratelimit import RetryableError
from apps.etl.runs import (
    fail_on_rejections,
    failing_on_termination,
    stop_on_streak,
    tracked_run,
)
from apps.market.models import DailyChip, DailyPrice, DailyValuation, Market, StockMaster

DATASET = "list_date"
MARKET_ROWS = (DailyPrice, DailyValuation, DailyChip)
UNSETTLED = (Outcome.EARLIER, Outcome.UNKNOWN)
Lookup = Callable[[str, date], list[date] | None]


class Command(BaseCommand):
    help = (
        "Find each stock's first TWSE or TPEx trading day and drop market rows before it or "
        "from delist_date on; run after the FinMind daily_price backfill."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--stock", action="append", dest="stocks")
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="also recheck stocks that already have a list_date",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        chosen = StockMaster.objects.order_by("stock_id")
        if options["stocks"]:
            chosen = chosen.filter(stock_id__in=options["stocks"])
        stocks = chosen if options["refresh"] else chosen.filter(list_date__isnull=True)
        firsts = _price_months(stocks)
        settled = _settled()
        todo = [
            stock
            for stock in stocks
            if stock.stock_id in firsts
            and (
                stock.list_date is not None
                or settled.get(stock.stock_id) != firsts[stock.stock_id][0].isoformat()
            )
        ]
        try:
            self._run(chosen, todo, firsts)
        except RejectedStocksError as exc:
            raise CommandError(str(exc)) from exc
        except RetryableError as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc

    def _run(
        self, chosen: QuerySet[StockMaster], todo: list[StockMaster], firsts: dict[str, list[date]]
    ) -> None:
        today = timezone.localdate()
        start = min((days[0] for days in firsts.values()), default=today)
        twse, tpex = TwseClient.build(limiter_for("twse")), TpexClient.build(limiter_for("tpex"))
        with failing_on_termination(), tracked_run("twse_tpex", DATASET, start, today) as run:
            run.checkpoint = {
                "delisting_dropped": self._drop_after_delisting(chosen),
                "listed": {},
                "rejected": {},
                "settled": {},
                "cleared": [],
            }
            streak = 0
            for stock in todo:
                probe = _probe(stock, twse, tpex)
                rejected = self._settle(run, stock, firsts[stock.stock_id], probe)
                streak = streak + 1 if rejected else 0
                stop_on_streak(streak, stock.stock_id)
            fail_on_rejections(run.checkpoint["rejected"])

    def _settle(self, run: EtlRun, stock: StockMaster, firsts: list[date], probe: Probe) -> bool:
        """Record one stock's outcome; True when it was rejected."""
        code, progress = stock.stock_id, run.checkpoint
        try:
            listing = first_listed_day(firsts, probe)
        except SourceError as exc:
            progress["rejected"][code] = f"{type(exc).__name__}: {exc}"
        else:
            if listing.outcome is Outcome.LISTED and listing.day is not None:
                progress["listed"][code] = [listing.day.isoformat(), self._set(stock, listing.day)]
                run.rows += 1
            elif listing.outcome is Outcome.MISSING:
                progress["rejected"][code] = str(listing.outcome)
            else:
                progress["settled"][code] = [firsts[0].isoformat(), str(listing.outcome)]
                if listing.outcome in UNSETTLED and stock.list_date is not None:
                    self._clear(stock)
                    progress["cleared"].append(code)
        run.save(update_fields=["checkpoint", "rows"])
        return code in progress["rejected"]

    def _set(self, stock: StockMaster, listed: date) -> int:
        with transaction.atomic():
            stock.list_date = listed
            stock.save(update_fields=["list_date"])
            dropped = _drop(Q(stock_id=stock.stock_id, date__lt=listed))
        self.stdout.write(f"{stock.stock_id}: listed {listed}, dropped {dropped} earlier rows")
        return dropped

    def _clear(self, stock: StockMaster) -> None:
        old, stock.list_date = stock.list_date, None
        stock.save(update_fields=["list_date"])
        self.stdout.write(
            f"{stock.stock_id}: listing date {old} no longer confirmed and cleared; rerun the "
            "FinMind backfill of daily_price, daily_valuation and daily_chip for it"
        )

    def _drop_after_delisting(self, stocks: QuerySet[StockMaster]) -> dict[str, int]:
        dropped = {}
        for stock in stocks.filter(delist_date__isnull=False):
            count = _drop(Q(stock_id=stock.stock_id, date__gte=stock.delist_date))
            if count:
                dropped[stock.stock_id] = count
                self.stdout.write(f"{stock.stock_id}: dropped {count} rows from delisting on")
        return dropped


def _drop(rows: Q) -> int:
    return sum(model.objects.filter(rows).delete()[0] for model in MARKET_ROWS)


def _price_months(stocks: QuerySet[StockMaster]) -> dict[str, list[date]]:
    """First traded day of every month with prices, per stock; traded means a close (SPEC §4)."""
    rows = (
        DailyPrice.objects.filter(stock_id__in=stocks.values("stock_id"), close__isnull=False)
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
    """First price day each settled stock was judged from; a new first price means a recheck."""
    found: dict[str, str] = {}
    for checkpoint in (
        EtlRun.objects.filter(dataset=DATASET)
        .order_by("run_id")
        .values_list("checkpoint", flat=True)
    ):
        found |= {code: entry[0] for code, entry in checkpoint.get("settled", {}).items()}
    return found


def _probe(stock: StockMaster, twse: TwseClient, tpex: TpexClient) -> Probe:
    """Current market first; every=True merges both; None when a needed market can't answer."""
    lookups: list[Lookup] = [twse.stock_month, tpex.stock_month]
    if stock.market == Market.TPEX:
        lookups.reverse()
    seen: dict[tuple[int, date], list[date] | None] = {}

    def ask(at: int, month: date) -> list[date] | None:
        if (at, month) not in seen:
            seen[at, month] = lookups[at](stock.stock_id, month)
        return seen[at, month]

    def probe(month: date, every: bool = False) -> list[date] | None:
        answers = []
        for at in range(len(lookups)):
            days = ask(at, month)
            if days and not every:
                return days
            answers.append(days)
        if None in answers:
            return None
        return sorted(day for days in answers for day in days or [])

    return probe
