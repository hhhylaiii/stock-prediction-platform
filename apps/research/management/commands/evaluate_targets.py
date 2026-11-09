"""Evaluate a return or interval run's research folds against its SPEC §7 baselines."""

from argparse import ArgumentParser
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import CalendarError, TradingDays
from apps.research.ensemble import write_report
from apps.research.holdout import is_holdout
from apps.research.labels import latest_price_date
from apps.research.prediction_cache import FoldMismatchError, MissingFoldsError
from apps.research.production import DESIGN_NAMES, FROZEN, design_for
from apps.research.target_eval import evaluate_run, report, scope_of
from apps.research.walkforward import WalkForwardError, generate_folds


class Command(BaseCommand):
    help = (
        "Report a return or interval run's research folds against the zero, trailing, "
        "volatility-band and logistic baselines; tuning folds are listed apart, holdout stays out."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run-key", required=True)
        parser.add_argument("--last-date")
        parser.add_argument("--output", help="Markdown file to write; prints when left out.")
        parser.add_argument(
            "--design",
            choices=DESIGN_NAMES,
            default=FROZEN,
            help="frozen: the walk-forward layout and up label production.DESIGNS holds; "
            "spec: the SPEC §7 defaults before the research-period comparison "
            "(36-month window, absolute up label).",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        run_key = options["run_key"]
        try:
            design = design_for(scope_of(run_key), options["design"])
            relative = design.target.relative_up
            last = as_date(options["last_date"], "--last-date") or latest_price_date()
            if last is None:
                raise ValueError("daily_price is empty")
            folds = [
                fold
                for fold in generate_folds(TradingDays.from_db(), last, design.config)
                if not is_holdout(fold)
            ]
            evaluated = evaluate_run(
                run_key, folds, settings.FEATURE_PARQUET_DIR, relative=relative
            )
            text = report(run_key, evaluated, relative)
            written = write_report(text, options["output"])
        except (
            ValueError,
            LookupError,
            OSError,
            CalendarError,
            WalkForwardError,
            FoldMismatchError,
            MissingFoldsError,
        ) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(text if written is None else f"wrote {written}")
