"""Write a day's predictions from each slot's active or degraded run (SPEC §7)."""

from argparse import ArgumentParser
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.research.labels import latest_price_date
from apps.research.lock import ResearchBusyError
from apps.research.predict import PredictionGateError, predict
from apps.research.runs import RunError


class Command(BaseCommand):
    help = (
        "Score every stock with features on --date with each slot's serving run, once the "
        "day's data quality gate passed; reruns replace the day's rows."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--date", help="Trading day to predict; the latest priced day if left out."
        )
        parser.add_argument(
            "--replace",
            action="store_true",
            help="Replace rows other runs already hold in a slot on the day.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            day = as_date(options["date"], "--date") or latest_price_date()
            if day is None:
                raise ValueError("daily_price is empty")
            result = predict(day, settings.FEATURE_PARQUET_DIR, options["replace"])
        except (
            PredictionGateError,
            RunError,
            ResearchBusyError,
            ValueError,
            FileNotFoundError,
        ) as error:
            raise CommandError(str(error)) from error
        for run_id, count in result.written.items():
            self.stdout.write(f"{run_id}: {count} rows")
        for slot in result.missing:
            self.stdout.write(f"no active or degraded run for {slot}")
