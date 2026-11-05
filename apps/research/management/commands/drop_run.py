"""Drop a research run that has no holdout claim and no model past candidate."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.research.runs import RunError, drop_run


class Command(BaseCommand):
    help = (
        "Delete a run's model_run rows, backtest metrics, model files and cached fold predictions."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("run_key")

    def handle(self, *args: Any, **options: Any) -> None:
        run_key = options["run_key"]
        try:
            dropped = drop_run(run_key)
        except (RunError, ValueError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(f"{run_key}: dropped {dropped} model_run rows")
