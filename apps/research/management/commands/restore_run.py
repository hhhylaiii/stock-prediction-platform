"""Put a degraded run back to active by hand, keeping its degraded days on record (SPEC §7)."""

from argparse import ArgumentParser
from datetime import date
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.market.calendar import CalendarError
from apps.research.lock import ResearchBusyError
from apps.research.promotion import ALERT_PERIODS, restore
from apps.research.runs import RunError


class Command(BaseCommand):
    help = (
        "Restore a degraded run to active from --date; [degraded_on, --date) joins its "
        "alert_periods with --reason."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run-id", required=True, help="The degraded run to restore.")
        parser.add_argument("--date", required=True, help="First active day, YYYY-MM-DD.")
        parser.add_argument("--reason", required=True, help="Why the run is restored.")

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            day = date.fromisoformat(options["date"])
            run = restore(options["run_id"], day, options["reason"])
        except (RunError, ResearchBusyError, CalendarError, ValueError) as error:
            raise CommandError(str(error)) from error
        period = run.params[ALERT_PERIODS][-1]
        self.stdout.write(
            f"{run.run_id} 自 {day} 恢復 active，警示期 {period['start']}–{period['end']}"
        )
