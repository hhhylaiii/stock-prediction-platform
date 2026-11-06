"""Evaluate a rank run's cached folds against Ridge into a report and backtest_metric (§7)."""

from argparse import ArgumentParser
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import CalendarError, TradingDays
from apps.research.baseline_runs import check_run_key
from apps.research.ensemble import write_report
from apps.research.holdout import is_holdout
from apps.research.labels import latest_price_date
from apps.research.models import HoldoutScope
from apps.research.prediction_cache import FoldMismatchError, MissingFoldsError
from apps.research.rank_eval import evaluate_run, report, write_metrics
from apps.research.walkforward import WalkForwardConfig, WalkForwardError, generate_folds

DEFAULT_CONFIG = WalkForwardConfig()


class Command(BaseCommand):
    help = (
        "Report a rank run's tuning and research folds against Ridge and store its "
        "backtest_metric rows; holdout folds stay out. train does not touch these rows, "
        "so run this again after refitting any fold of the run_key."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run-key", required=True)
        parser.add_argument("--last-date")
        parser.add_argument("--output", help="Markdown file to write; prints when left out.")

    def handle(self, *args: Any, **options: Any) -> None:
        run_key = options["run_key"]
        try:
            check_run_key(HoldoutScope.RANK, run_key)
            last = as_date(options["last_date"], "--last-date") or latest_price_date()
            if last is None:
                raise ValueError("daily_price is empty")
            calendar = TradingDays.from_db()
            folds = [
                fold
                for fold in generate_folds(calendar, last, DEFAULT_CONFIG)
                if not is_holdout(fold)
            ]
            evaluations = evaluate_run(run_key, folds, settings.FEATURE_PARQUET_DIR, calendar)
            text = report(run_key, evaluations)
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
        stored = write_metrics(run_key, evaluations)
        self.stdout.write(text if written is None else f"wrote {written}")
        self.stdout.write(f"{run_key}: {stored} backtest_metric rows")
