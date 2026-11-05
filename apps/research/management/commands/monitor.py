from datetime import date, timedelta
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.utils import timezone

from apps.market.calendar import CalendarError, TradingDays
from apps.research.monitor import MonitorError, run_monitor


class Command(BaseCommand):
    help = "Score predictions maturing on a trading day and check that day's outputs."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--date", help="Trading day, YYYY-MM-DD; defaults to the last one before today."
        )

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            calendar = TradingDays.from_db()
            raw = options.get("date")
            day = (
                date.fromisoformat(str(raw))
                if raw
                else calendar.latest_on_or_before(timezone.localdate() - timedelta(days=1))
            )
            result = run_monitor(day, calendar)
        except (CalendarError, MonitorError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(
            f"monitor {day}: {result.scored} model_monitor rows, {len(result.logs)} quality rows"
        )
