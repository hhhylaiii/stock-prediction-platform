"""Rebuild label_daily for a range, or for the most recent trading days (SPEC §6 處理步驟 6)."""

from argparse import ArgumentParser
from collections.abc import Iterator
from datetime import date
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.market.calendar import TradingDays
from apps.research.labels import build_labels, latest_price_date, write_labels

RECENT_DAYS = 60


def _years(start: date, end: date) -> Iterator[tuple[date, date]]:
    for year in range(start.year, end.year + 1):
        yield max(start, date(year, 1, 1)), min(end, date(year, 12, 31))


class Command(BaseCommand):
    help = "Rebuild label_daily year by year; without --start it redoes the last 60 trading days."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--start", type=date.fromisoformat)
        parser.add_argument("--end", type=date.fromisoformat)
        parser.add_argument("--recent", type=int, default=RECENT_DAYS)

    def handle(self, *args: Any, **options: Any) -> None:
        last = latest_price_date()
        if last is None:
            raise CommandError("daily_price is empty")
        calendar = TradingDays.from_db()
        end = min(options["end"] or last, last)
        start = options["start"] or calendar.shift(
            calendar.latest_on_or_before(end), 1 - options["recent"]
        )
        if start > end:
            raise CommandError(f"--start {start} is after the end {end}")
        rows = sum(
            write_labels(build_labels(calendar, low, high), low, high)
            for low, high in _years(start, end)
        )
        self.stdout.write(f"label_daily: {rows} rows for {start}..{end}")
