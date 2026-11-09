"""Add one second-batch feature group to the rank design on coarse research folds."""

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
    parse_seeds,
    stopping_rounds,
    train_arms,
    window_arms,
)
from apps.research.feature_groups import (
    BASES,
    FEATURE_GROUPS,
    base_arm,
    blocked_lines,
    check_recorded_screening,
    check_same_ics,
    group_experiment,
    group_names,
    group_screening,
    screening_lines,
    with_features,
)
from apps.research.holdout import HoldoutError, holdout_claimed
from apps.research.labels import latest_price_date
from apps.research.models import HoldoutScope
from apps.research.prediction_cache import FoldMismatchError, MissingFoldsError
from apps.research.runs import RunError
from apps.research.walkforward import WalkForwardConfig, WalkForwardError

DEFAULT_CONFIG = WalkForwardConfig()


def _tags(text: str | None) -> list[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]


class Command(BaseCommand):
    help = (
        "Screen the rank features again with one second-batch group (after the accepted ones), "
        "fit the design with and without it per seed, and report whether the rank IC improves."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--group", required=True, choices=list(FEATURE_GROUPS))
        parser.add_argument("--accepted", help="Groups kept earlier, comma separated, in order.")
        parser.add_argument("--base", default="floor", choices=list(BASES))
        parser.add_argument("--window", default="w60", choices=list(window_arms(DEFAULT_CONFIG)))
        parser.add_argument("--run-key", required=True)
        parser.add_argument("--last-date")
        parser.add_argument("--seeds", default=SUGGESTED_SEEDS)
        parser.add_argument("--note", help="Caveat printed at the top of the report.")
        parser.add_argument("--output", help="Markdown file to write; prints when left out.")

    def handle(self, *args: Any, **options: Any) -> None:
        if holdout_claimed(HoldoutScope.RANK):
            raise CommandError("a rank holdout run exists, so the rank design is frozen (SPEC §7)")
        group, accepted, run_key = options["group"], _tags(options["accepted"]), options["run_key"]
        directory = settings.FEATURE_PARQUET_DIR
        try:
            names = group_names(accepted, group)
            seeds = parse_seeds(options["seeds"])
            start = base_arm(DEFAULT_CONFIG, options["window"], options["base"])
            tags = [*accepted, group]
            for probe in (with_features(start, accepted, ()), with_features(start, tags, ())):
                for seed in seeds:
                    check_run_key(HoldoutScope.RANK, arm_key(run_key, probe, seed))
            last = as_date(options["last_date"], "--last-date") or latest_price_date()
            if last is None:
                raise ValueError("daily_price is empty")
            calendar = TradingDays.from_db()
            recorded = check_recorded_screening(directory)
            kept_before = lgbm_runs.scope_features(HoldoutScope.RANK)
            accepted_kept = None
            if accepted:
                before = group_screening(group_names(accepted), directory, calendar, last)
                check_same_ics(before, recorded)
                accepted_kept = kept_before = before.kept
            screened = group_screening(names, directory, calendar, last)
            if not accepted:
                check_same_ics(screened, recorded)
            lines = screening_lines(screened, group, kept_before)
            added = [name for name in FEATURE_GROUPS[group] if name in screened.kept]
            if not added:
                verdict = "**判定：這組在篩選全部剔除，不跑 walk-forward，維持原設計**"
                text = "\n".join(
                    [f"# 特徵組：{group}（{run_key}）", "", *lines, verdict, "", *blocked_lines()]
                )
                written = write_report(text, options["output"])
                self.stdout.write(text if written is None else f"wrote {written}")
                return
            experiment = group_experiment(start, accepted, accepted_kept, group, screened.kept)
            folds = compared_folds(experiment, calendar, last)
            first = folds[experiment.arms[0].name][0]
            shuffled = checked_shuffle(first, directory, seeds[0])
            self.stdout.write(f"fold {first.index}: shuffled OOS rank IC {shuffled:.4f}")
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
        text = "\n".join([text.rstrip("\n"), "", *lines, *blocked_lines()])
        written = write_report(text, options["output"])
        self.stdout.write(text if written is None else f"wrote {written}")
