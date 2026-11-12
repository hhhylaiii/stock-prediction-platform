"""Check live feature coverage for a day or a range, writing its data_quality_log rows."""

from datetime import date, timedelta
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.utils import timezone

from apps.market.calendar import CalendarError, TradingDays
from apps.research.feature_coverage import CoverageError, coverage_summary, run_coverage
from apps.research.lock import ResearchBusyError


class Command(BaseCommand):
    help = "Compare each day's universe feature coverage with its reference months; a row a day."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--date",
            type=date.fromisoformat,
            help="Trading day, YYYY-MM-DD; defaults to the last one before today.",
        )
        parser.add_argument("--start", type=date.fromisoformat, help="First day of a range.")
        parser.add_argument("--end", type=date.fromisoformat, help="Last day of a range.")
        parser.add_argument(
            "--quiet", action="store_true", help="Write the rows without alerting, as a backfill."
        )

    def _days(self, calendar: TradingDays, options: dict[str, Any]) -> tuple[date, ...]:
        day, start, end = options.get("date"), options.get("start"), options.get("end")
        if day is not None and (start is not None or end is not None):
            raise CommandError("--date cannot be combined with --start or --end")
        if start is None or end is None:
            if start is not None or end is not None:
                raise CommandError("--start and --end go together")
            yesterday = timezone.localdate() - timedelta(days=1)
            return (day or calendar.latest_on_or_before(yesterday),)
        return calendar.between(start, end)

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            calendar = TradingDays.from_db()
            for day in self._days(calendar, options):
                row = run_coverage(
                    day, calendar, settings.FEATURE_PARQUET_DIR, alert=not options.get("quiet")
                )
                self.stdout.write(
                    f"feature coverage {day}: {row.severity}, {coverage_summary(row)}"
                )
        except (CalendarError, CoverageError, ResearchBusyError) as exc:
            raise CommandError(str(exc)) from exc
