"""Refit the LightGBM rank folds with XGBoost to catch implementation errors (SPEC §7)."""

from argparse import ArgumentParser
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import CalendarError
from apps.research.baseline_runs import check_run_key
from apps.research.ensemble import research_folds, write_report
from apps.research.management.commands.train import fold_indices
from apps.research.models import HoldoutScope
from apps.research.production import DESIGN_NAMES, FROZEN, design_for
from apps.research.walkforward import WalkForwardError
from apps.research.xgboost_check import xgboost_daily_ics, xgboost_report


class Command(BaseCommand):
    help = "Compare XGBoost refits with the recorded LightGBM rank IC on research folds."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run-key", required=True)
        parser.add_argument("--folds", help="Research fold indices like 4-9,12; all when left out.")
        parser.add_argument("--last-date")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--output", help="Markdown file to write; prints when left out.")
        parser.add_argument(
            "--design",
            choices=DESIGN_NAMES,
            default=FROZEN,
            help="frozen: the walk-forward layout production.DESIGNS holds; "
            "spec: the SPEC §7 defaults before the research-period comparison (36-month window).",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        run_key = options["run_key"]
        try:
            check_run_key(HoldoutScope.RANK, run_key)
            last = as_date(options["last_date"], "--last-date")
            design = design_for(HoldoutScope.RANK, options["design"])
            folds = research_folds(design.config, last)
            if options["folds"] is not None:
                wanted = fold_indices(options["folds"])
                outside = sorted(wanted - {fold.index for fold in folds})
                if outside:
                    raise ValueError(f"folds {outside} are not research folds")
                folds = [fold for fold in folds if fold.index in wanted]
            daily = xgboost_daily_ics(
                run_key,
                folds,
                settings.FEATURE_PARQUET_DIR,
                options["seed"],
                design.variant.min_rounds,
            )
        except (
            ValueError,
            LookupError,
            CalendarError,
            WalkForwardError,
            RuntimeError,
            FileNotFoundError,
        ) as error:
            raise CommandError(str(error)) from error
        text = xgboost_report(run_key, daily)
        written = write_report(text, options["output"])
        self.stdout.write(text if written is None else f"wrote {written}")
