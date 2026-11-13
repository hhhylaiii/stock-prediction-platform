"""Rebuild label_daily for a range, or for the most recent trading days (SPEC §6 處理步驟 6)."""

from argparse import ArgumentParser
from datetime import date
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.market.calendar import TradingDays
from apps.research.labels import build_labels, latest_price_date, write_labels
from apps.research.lock import ResearchBusyError, research_lock
from apps.research.ranges import years

RECENT_DAYS = 60


class Command(BaseCommand):
    help = (
        "Rebuild label_daily year by year; without --start it redoes the last 60 trading days. "
        "Holds the research lock for the whole run and fails at once when another job holds it."
    )

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
        try:
            with research_lock(f"build_labels {start}..{end}"):
                rows = sum(
                    write_labels(build_labels(calendar, low, high), low, high)
                    for low, high in years(start, end)
                )
        except ResearchBusyError as error:
            raise CommandError(str(error)) from error
        self.stdout.write(f"label_daily: {rows} rows for {start}..{end}")
