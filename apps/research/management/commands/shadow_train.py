"""Fit and freeze a shadow candidate on its slot's serving fold (holding-shadow-prereg.md §7)."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.market.calendar import CalendarError
from apps.research.candidates import freeze, measure
from apps.research.holdout import HoldoutError
from apps.research.lock import ResearchBusyError
from apps.research.models import ShadowCandidate
from apps.research.runs import RunError


class Command(BaseCommand):
    help = (
        "Fit a shadow candidate on the production fold of its slot's serving run and freeze it: "
        "the model file under MODEL_DIR/shadow and one shadow_artifact row. --output only fits "
        "into a folder and prints its time, memory and sha256, writing nothing to the database."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--candidate", required=True, choices=ShadowCandidate.values)
        parser.add_argument(
            "--replace",
            action="store_true",
            help="Freeze the next generation on a serving run that already has one.",
        )
        parser.add_argument("--output", type=Path, help="Fit into this folder only.")

    def handle(self, *args: Any, **options: Any) -> None:
        name, output = options["candidate"], options["output"]
        directory = Path(settings.FEATURE_PARQUET_DIR)
        try:
            if output is not None:
                if options["replace"]:
                    raise ValueError("--replace freezes; --output only fits")
                measured = measure(name, directory, output)
            else:
                row = freeze(name, directory, options["replace"])
        except (
            ValueError,
            RunError,
            HoldoutError,
            ResearchBusyError,
            FileNotFoundError,
            RuntimeError,
            CalendarError,
        ) as error:
            raise CommandError(str(error)) from error
        if output is not None:
            self.stdout.write(
                f"{name}: {measured.path} sha256 {measured.sha256}, "
                f"{measured.seconds} s, peak {measured.peak_mb} MB"
            )
            return
        recipe = row.recipe
        self.stdout.write(
            f"{name} g{row.generation} on {row.serving_run_id}: {row.path} sha256 {row.sha256}, "
            f"first signal {row.first_signal}, {recipe['seconds']} s, peak {recipe['peak_mb']} MB"
        )
