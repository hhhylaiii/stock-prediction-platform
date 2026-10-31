"""Run one source dataset for a date or a date range: fetch to raw, normalize, upsert."""

from argparse import ArgumentParser
from datetime import date
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.etl.adapters import build_adapter
from apps.etl.adapters.base import SourceError
from apps.etl.management.options import as_date
from apps.etl.ratelimit import RetryableError
from apps.market.calendar import CalendarError


class Command(BaseCommand):
    help = "Fetch a dataset from a source into the raw and standard tables."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--source", required=True)
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--date")
        parser.add_argument("--start")
        parser.add_argument("--end")

    def handle(self, *args: Any, **options: Any) -> None:
        start, end = self._date_range(
            as_date(options["date"], "--date"),
            as_date(options["start"], "--start"),
            as_date(options["end"], "--end"),
        )
        try:
            adapter = build_adapter(options["source"])
            run = adapter.run(options["dataset"], start, end)
        except (ValueError, CalendarError, SourceError, RetryableError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        self.stdout.write(
            f"etl_run {run.run_id} {run.status}: {run.rows} rows "
            f"{options['source']}/{options['dataset']} {start}..{end}"
        )

    @staticmethod
    def _date_range(single: date | None, start: date | None, end: date | None) -> tuple[date, date]:
        if single is not None:
            if start is not None or end is not None:
                raise CommandError("use either --date or --start/--end, not both")
            return single, single
        if start is None or end is None:
            raise CommandError("give --date, or both --start and --end")
        if start > end:
            raise CommandError(f"--start {start} is after --end {end}")
        return start, end
