"""Fetch the trading days the daily datasets missed and check those days again."""

from argparse import ArgumentParser
from datetime import timedelta
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.etl.catchup import NOTHING, catch_up, plan_catch_up, select_jobs
from apps.etl.management.options import as_date, required_date
from apps.market.calendar import CalendarError, TradingDays


class Command(BaseCommand):
    help = "Fetch missed trading days of the daily TWSE, TPEx and Yahoo datasets."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--since", required=True)
        parser.add_argument(
            "--until", help="last day to look at; defaults to the last session before today"
        )
        parser.add_argument("--source")
        parser.add_argument("--dataset")
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="list the days each dataset would fetch, fetch nothing",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        since = required_date(options["since"], "--since")
        until = as_date(options["until"], "--until")
        jobs = select_jobs(options["source"], options["dataset"])
        if not jobs:
            raise CommandError(
                f"no catch-up job for source {options['source']!r} dataset {options['dataset']!r}"
            )
        try:
            calendar = TradingDays.from_db()
            until = until or calendar.latest_on_or_before(timezone.localdate() - timedelta(days=1))
            if since > until:
                raise CommandError(f"--since {since} must not be after --until {until}")
            if options["dry_run"]:
                self._print_plan(plan_catch_up(since, until, calendar, jobs))
                return
            report = catch_up(since, until, calendar, jobs)
        except CalendarError as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        for line in report.lines():
            self.stdout.write(line)
        if report.failed:
            raise CommandError(f"{len(report.failed)} catch-up step(s) failed")

    def _print_plan(self, planned: dict[str, list[str]]) -> None:
        for key, days in planned.items():
            self.stdout.write(f"to fetch {key}: {', '.join(days)}")
        if not planned:
            self.stdout.write(NOTHING)
