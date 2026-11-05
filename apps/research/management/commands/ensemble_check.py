"""Compare the rank-average ensemble of Ridge and LightGBM with LightGBM alone (SPEC §7)."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import CalendarError
from apps.research.baseline_runs import check_run_key
from apps.research.comparisons import ensemble_ics, ensemble_report, research_folds, write_report
from apps.research.labels import HORIZONS
from apps.research.models import HoldoutScope
from apps.research.prediction_cache import FoldMismatchError, MissingFoldsError
from apps.research.walkforward import WalkForwardConfig, WalkForwardError

DEFAULT_CONFIG = WalkForwardConfig()


class Command(BaseCommand):
    help = "Report research rank IC of Ridge, LightGBM and their rank average from cached folds."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run-key", required=True)
        parser.add_argument("--last-date")
        parser.add_argument("--output", help="Markdown file to write; prints when left out.")

    def handle(self, *args: Any, **options: Any) -> None:
        run_key = options["run_key"]
        try:
            check_run_key(HoldoutScope.RANK, run_key)
            last = as_date(options["last_date"], "--last-date")
            folds = research_folds(DEFAULT_CONFIG, last)
            ics = {horizon: ensemble_ics(run_key, folds, horizon) for horizon in HORIZONS}
        except (
            ValueError,
            CalendarError,
            WalkForwardError,
            MissingFoldsError,
            FoldMismatchError,
        ) as error:
            raise CommandError(str(error)) from error
        text = ensemble_report(run_key, ics)
        written = write_report(text, options["output"])
        self.stdout.write(text if written is None else f"wrote {written}")
