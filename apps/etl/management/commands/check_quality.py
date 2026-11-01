"""Run the daily data-quality checks for one trading day and send alerts."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.etl.quality import run_checks
from apps.market.calendar import CalendarError, TradingDays


class Command(BaseCommand):
    help = "Check one trading day's standard tables and record the results in data_quality_log."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--date", required=True)

    def handle(self, *args: Any, **options: Any) -> None:
        day = as_date(options["date"], "--date")
        if day is None:
            raise CommandError("give --date")
        try:
            calendar = TradingDays.from_db()
            if not calendar.is_trading_day(day):
                raise CommandError(f"{day} is not a trading day")
        except CalendarError as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        results = run_checks(day, calendar)
        for row in results:
            self.stdout.write(
                f"{day} {row.check_name} {row.severity} {row.detail.get('table', '')}"
            )
