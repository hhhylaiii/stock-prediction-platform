"""Run one SPEC §7 rank experiment on coarse research folds and report its verdict."""

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
    experiment_report,
    experiments,
    parse_seeds,
    stopping_rounds,
    train_arms,
    tuning_rounds,
    with_tuned_rounds,
)
from apps.research.holdout import HoldoutError, holdout_claimed
from apps.research.labels import latest_price_date
from apps.research.models import HoldoutScope
from apps.research.prediction_cache import FoldMismatchError, MissingFoldsError
from apps.research.runs import RunError
from apps.research.walkforward import WalkForwardConfig, WalkForwardError

DEFAULT_CONFIG = WalkForwardConfig()


class Command(BaseCommand):
    help = (
        "Fit every arm of a rank experiment on six-month research folds once per seed, each "
        "under <run-key>-<arm>[-s<seed>], and report the seed-averaged rank IC with the verdict."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--name", required=True, choices=sorted(experiments(DEFAULT_CONFIG)))
        parser.add_argument("--run-key", required=True)
        parser.add_argument("--last-date")
        parser.add_argument("--seeds", default="0", help=f"Distinct seeds, e.g. {SUGGESTED_SEEDS}.")
        parser.add_argument("--note", help="Caveat printed at the top of the report.")
        parser.add_argument("--output", help="Markdown file to write; prints when left out.")

    def handle(self, *args: Any, **options: Any) -> None:
        experiment = experiments(DEFAULT_CONFIG)[options["name"]]
        run_key = options["run_key"]
        if holdout_claimed(HoldoutScope.RANK):
            raise CommandError("a rank holdout run exists, so the rank design is frozen (SPEC §7)")
        try:
            seeds = parse_seeds(options["seeds"])
            for arm in experiment.arms:
                for seed in seeds:
                    check_run_key(HoldoutScope.RANK, arm_key(run_key, arm, seed))
            lgbm_runs.scope_features(HoldoutScope.RANK)
            last = as_date(options["last_date"], "--last-date") or latest_price_date()
            if last is None:
                raise ValueError("daily_price is empty")
            calendar = TradingDays.from_db()
            folds = compared_folds(experiment, calendar, last)
            directory = settings.FEATURE_PARQUET_DIR
            first = folds[experiment.arms[0].name][0]
            shuffled = checked_shuffle(first, directory, seeds[0])
            self.stdout.write(f"fold {first.index}: shuffled OOS rank IC {shuffled:.4f}")
            if any(arm.tuned for arm in experiment.arms):
                rounds = tuning_rounds(experiment, run_key, calendar, last, directory)
                self.stdout.write(f"tuned rounds from the tuning folds: {rounds}")
                experiment = with_tuned_rounds(experiment, rounds)
            ran = train_arms(experiment, run_key, folds, directory, seeds)
            daily = daily_ics(experiment, run_key, folds, seeds)
            stopped = stopping_rounds(experiment, run_key, folds, seeds)
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
        for name, indices in ran.items():
            self.stdout.write(f"{name}: ran folds {indices}")
        text = experiment_report(experiment, run_key, daily, options["note"], stopped)
        written = write_report(text, options["output"])
        self.stdout.write(text if written is None else f"wrote {written}")
