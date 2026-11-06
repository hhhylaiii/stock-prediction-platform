"""Refit rank without the feature groups delisted stocks lack and report what changes (K17)."""

from argparse import ArgumentParser
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import CalendarError, TradingDays
from apps.research import lgbm_runs
from apps.research.baseline_runs import check_run_key
from apps.research.ensemble import write_report
from apps.research.experiments import (
    SUGGESTED_SEEDS,
    arm_key,
    checked_shuffle,
    compared_folds,
    daily_ics,
    parse_seeds,
    train_arms,
    window_arms,
)
from apps.research.holdout import HoldoutError, holdout_claimed
from apps.research.labels import latest_price_date
from apps.research.missing_mask import (
    delisting_scores,
    delisting_windows,
    mask_experiment,
    mask_report,
    masked_gain_share,
    masked_values,
    missing_split,
)
from apps.research.models import HoldoutScope
from apps.research.prediction_cache import FoldMismatchError, MissingFoldsError
from apps.research.runs import RunError
from apps.research.walkforward import WalkForwardConfig, WalkForwardError

DEFAULT_CONFIG = WalkForwardConfig()


class Command(BaseCommand):
    help = (
        "Fit a rank window and the same window with the delisting-prone features masked, once "
        "per seed under <run-key>-<arm>[-s<seed>], and report the IC, the scores of stocks "
        "about to delist and the gain the masked features held."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run-key", required=True)
        parser.add_argument(
            "--window",
            default="base",
            choices=list(window_arms(DEFAULT_CONFIG)),
            help="Training window both arms train on; use the window experiment's winner.",
        )
        parser.add_argument("--last-date")
        parser.add_argument("--seeds", default="0", help=f"Distinct seeds, e.g. {SUGGESTED_SEEDS}.")
        parser.add_argument("--note", help="Caveat printed at the top of the report.")
        parser.add_argument("--output", help="Markdown file to write; prints when left out.")

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            experiment = mask_experiment(DEFAULT_CONFIG, options["window"])
        except ValueError as error:
            raise CommandError(str(error)) from error
        run_key = options["run_key"]
        if holdout_claimed(HoldoutScope.RANK):
            raise CommandError("a rank holdout run exists, so the rank design is frozen (SPEC §7)")
        plain, masked = experiment.arms
        try:
            seeds = parse_seeds(options["seeds"])
            for arm in experiment.arms:
                for seed in seeds:
                    check_run_key(HoldoutScope.RANK, arm_key(run_key, arm, seed))
            screened = lgbm_runs.scope_features(HoldoutScope.RANK)
            last = as_date(options["last_date"], "--last-date") or latest_price_date()
            if last is None:
                raise ValueError("daily_price is empty")
            calendar = TradingDays.from_db()
            folds = compared_folds(experiment, calendar, last)
            directory = settings.FEATURE_PARQUET_DIR
            first = folds[plain.name][0]
            shuffled = checked_shuffle(first, directory, seeds[0])
            self.stdout.write(f"fold {first.index}: shuffled OOS rank IC {shuffled:.4f}")
            ran = train_arms(experiment, run_key, folds, directory, seeds)
            daily = daily_ics(experiment, run_key, folds, seeds)
            scores = delisting_scores(
                experiment, run_key, folds, seeds, delisting_windows(calendar)
            )
            kept = [name for name in masked.variant.masked if name in screened]
            split = None
            if kept and not scores.empty:
                split = missing_split(experiment, scores, masked_values(scores, directory, kept))
            shares = masked_gain_share(experiment, run_key, folds, seeds)
        except (
            ValueError,
            LookupError,
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
        for name, indices in ran.items():
            self.stdout.write(f"{name}: ran folds {indices}")
        text = mask_report(experiment, run_key, daily, scores, split, shares, kept, options["note"])
        written = write_report(text, options["output"])
        self.stdout.write(text if written is None else f"wrote {written}")
