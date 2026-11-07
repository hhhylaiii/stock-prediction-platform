"""Fit a scope's production model N and its shadow N′ on the frozen design (SPEC §7)."""

from argparse import ArgumentParser
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import CalendarError
from apps.research.holdout import HoldoutError
from apps.research.labels import latest_price_date
from apps.research.models import HoldoutScope
from apps.research.monitor import BRIER_BASELINE, IC_THRESHOLD
from apps.research.prediction_cache import FoldMismatchError, MissingFoldsError
from apps.research.production import build_run
from apps.research.runs import RunError
from apps.research.walkforward import WalkForwardError


class Command(BaseCommand):
    help = (
        "Fit a scope's production model through --as-of and its shadow on the months before, "
        "with the shadow's baselines for return and interval, as candidates; only once the "
        "scope's holdout has been claimed."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--scope", required=True, choices=HoldoutScope.values)
        parser.add_argument("--run-key", required=True)
        parser.add_argument(
            "--research-run-key",
            required=True,
            help="The frozen design's walk-forward run, whose folds give the monitor references.",
        )
        parser.add_argument("--as-of", help="Latest priced day the model trains up to.")
        parser.add_argument("--seed", type=int, default=0)

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            as_of = as_date(options["as_of"], "--as-of") or latest_price_date()
            if as_of is None:
                raise ValueError("daily_price is empty")
            built = build_run(
                options["scope"],
                options["run_key"],
                options["research_run_key"],
                as_of,
                settings.FEATURE_PARQUET_DIR,
                options["seed"],
            )
        except (
            ValueError,
            RunError,
            HoldoutError,
            FileNotFoundError,
            RuntimeError,
            CalendarError,
            WalkForwardError,
            FoldMismatchError,
            MissingFoldsError,
        ) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(f"{options['run_key']}: {', '.join(built.model)}")
        self.stdout.write(f"{options['run_key']}-shadow: {', '.join(built.shadow)}")
        source = options["research_run_key"]
        if IC_THRESHOLD in built.references:
            listed = ", ".join(
                f"{h} 日 {value:.4f}" for h, value in built.references[IC_THRESHOLD].items()
            )
            self.stdout.write(f"rolling IC thresholds from {source}: {listed}")
        if BRIER_BASELINE in built.references:
            self.stdout.write(
                f"Brier baseline from {source}: {built.references[BRIER_BASELINE]:.4f}"
            )
