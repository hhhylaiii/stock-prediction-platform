"""Add the bars chosen stocks lack on one day, from the exchange's quotes of that day."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.etl.errors import SourceError
from apps.etl.management.options import required_date
from apps.etl.price_fill import FILLABLE, Bar, fill_price_bars
from apps.etl.ratelimit import RetryableError
from apps.market.calendar import TradingDays

PRICES = ("open", "high", "low", "close")
COUNTS = ("volume", "turnover")


class Command(BaseCommand):
    help = (
        "Insert the daily_price bars chosen stocks lack on a day before the first kept quote "
        "page; the answers are kept as daily_price_fill, never as that day's quotes."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--source", required=True, choices=FILLABLE)
        parser.add_argument("--date", required=True)
        parser.add_argument("--stock", action="append", dest="stocks", required=True)
        parser.add_argument("--dry-run", action="store_true", help="print the bars; nothing stays")

    def handle(self, *args: Any, **options: Any) -> None:
        day = required_date(options["date"], "--date")
        dry_run = bool(options["dry_run"])
        source, stocks = options["source"], options["stocks"]
        try:
            calendar = TradingDays.from_db()
            if dry_run:
                with transaction.atomic():
                    bars = fill_price_bars(source, day, stocks, calendar)
                    transaction.set_rollback(True)
            else:
                bars = fill_price_bars(source, day, stocks, calendar)
        except (ValueError, SourceError, RetryableError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        for bar in bars:
            self.stdout.write(_line(bar))
        verb = "to add" if dry_run else "added"
        self.stdout.write(f"daily_price: {len(bars)} bars {verb} for {day}")
        if dry_run:
            self.stdout.write("nothing was written")


def _line(bar: Bar) -> str:
    figures = " ".join(f"{name} {bar[name]}" for name in (*PRICES, *COUNTS))
    return f"insert {bar['stock_id']} {bar['date']} {figures}"
