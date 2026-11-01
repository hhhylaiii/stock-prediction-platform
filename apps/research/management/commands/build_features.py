"""Rebuild feature_daily and its Parquet for a range, a day, or the recent days (SPEC §6)."""

from argparse import ArgumentParser
from datetime import date
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.market.calendar import TradingDays
from apps.research.features.build import build_feature_frame
from apps.research.features.store import export_year, write_features
from apps.research.labels import latest_price_date
from apps.research.ranges import years

RECENT_DAYS = 60


class Command(BaseCommand):
    help = "Rebuild feature_daily year by year; without a range it redoes the last 60 trading days."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--start", type=date.fromisoformat)
        parser.add_argument("--end", type=date.fromisoformat)
        parser.add_argument("--date", type=date.fromisoformat)
        parser.add_argument("--recent", type=int, default=RECENT_DAYS)

    def _range(self, calendar: TradingDays, options: dict[str, Any]) -> tuple[date, date]:
        last = latest_price_date()
        if last is None:
            raise CommandError("daily_price is empty")
        for name in ("date", "start", "end"):
            given = options[name]
            if given is not None and not calendar.start <= given <= calendar.end:
                raise CommandError(f"--{name} {given} is outside the trading calendar")
        day = options["date"]
        if day is not None:
            if options["start"] or options["end"]:
                raise CommandError("--date cannot be combined with --start or --end")
            if not calendar.is_trading_day(day):
                raise CommandError(f"{day} is not a trading day")
            if day > last:
                raise CommandError(f"{day} is after the last price day {last}")
            return day, day
        end = min(options["end"] or last, last)
        start = options["start"] or calendar.shift(
            calendar.latest_on_or_before(end), 1 - options["recent"]
        )
        if start > end:
            raise CommandError(f"--start {start} is after the end {end}")
        return start, end

    def handle(self, *args: Any, **options: Any) -> None:
        calendar = TradingDays.from_db()
        start, end = self._range(calendar, options)
        rows = 0
        for low, high in years(start, end):
            rows += write_features(build_feature_frame(calendar, low, high), low, high)
            export_year(low.year, settings.FEATURE_PARQUET_DIR)
        self.stdout.write(f"feature_daily: {rows} rows for {start}..{end}")
