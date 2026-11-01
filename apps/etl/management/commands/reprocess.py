"""Rebuild one dataset's standard rows from its stored raw responses, without refetching."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.etl.errors import SourceError
from apps.etl.management.options import as_date
from apps.etl.replay import reprocess
from apps.market.calendar import CalendarError, TradingDays


class Command(BaseCommand):
    help = "Rebuild standard rows of a TWSE or TPEx dataset from raw for a date range."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--source", required=True)
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--start", required=True)
        parser.add_argument("--end", required=True)

    def handle(self, *args: Any, **options: Any) -> None:
        start = as_date(options["start"], "--start")
        end = as_date(options["end"], "--end")
        if start is None or end is None or start > end:
            raise CommandError(f"--start {start} must not be after --end {end}")
        try:
            run = reprocess(
                options["source"], options["dataset"], start, end, TradingDays.from_db()
            )
        except (ValueError, CalendarError, SourceError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        self.stdout.write(
            f"etl_run {run.run_id} {run.status}: {run.rows} rows rebuilt "
            f"{options['source']}/{options['dataset']} {start}..{end}"
        )
