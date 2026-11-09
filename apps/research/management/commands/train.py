"""Fit baselines or LightGBM fold by fold for a target family, or run the shuffle check."""

import math
from argparse import ArgumentParser
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import CalendarError, TradingDays
from apps.research import lgbm_runs
from apps.research.baseline_runs import (
    BASELINE,
    SCOPES,
    check_run_key,
    run_folds,
    scope_features,
    shuffle_check,
)
from apps.research.gbdt import bag_seeds
from apps.research.holdout import HoldoutError, is_holdout
from apps.research.labels import latest_price_date
from apps.research.models import HoldoutScope
from apps.research.production import DESIGN_NAMES, FROZEN, Design, design_for
from apps.research.runs import RunError
from apps.research.walkforward import Fold, WalkForwardError, generate_folds

MODELS = (BASELINE, lgbm_runs.MODEL)


def fold_indices(text: str) -> set[int]:
    """Indices from a list like 0-3,7."""
    chosen: set[int] = set()
    for part in text.split(","):
        low, dash, high = part.strip().partition("-")
        try:
            first, last = int(low), int(high if dash else low)
        except ValueError:
            raise CommandError(f"--folds takes indices like 0-3,7, got {text!r}") from None
        if first > last:
            raise CommandError(f"--folds takes indices like 0-3,7, got {text!r}")
        chosen.update(range(first, last + 1))
    return chosen


class Command(BaseCommand):
    help = (
        "Fit one scope's baselines or LightGBM on research folds, caching predictions; "
        "or --shuffle-check."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--model", choices=MODELS, default=BASELINE)
        parser.add_argument("--scope", choices=list(SCOPES))
        parser.add_argument("--run-key")
        parser.add_argument("--folds")
        parser.add_argument("--last-date")
        parser.add_argument(
            "--rerun",
            action="store_true",
            help="Refit cached folds too. A model's cache is reused while its fold boundaries, "
            "feature partitions, labels and recipe are unchanged; other models' caches stay.",
        )
        parser.add_argument("--shuffle-check", action="store_true")
        parser.add_argument(
            "--seed", type=int, default=0, help="Seed of the shuffle check and of LightGBM."
        )
        parser.add_argument(
            "--design",
            choices=DESIGN_NAMES,
            default=FROZEN,
            help="frozen: the walk-forward layout and variants production.DESIGNS holds; "
            "spec: the SPEC §7 defaults before the research-period comparison (36-month window).",
        )
        parser.add_argument(
            "--bags",
            type=int,
            default=1,
            help="LightGBM seeds averaged per model: 1 for research folds, 5 for final runs "
            "under their own run_key, since a run_key keeps one bag size.",
        )

    def _design(self, options: dict[str, Any]) -> Design:
        """The scope's chosen design; the shuffle check alone reads the rank layout."""
        return design_for(options["scope"] or HoldoutScope.RANK, options["design"])

    def _folds(self, options: dict[str, Any]) -> list[Fold]:
        last = as_date(options["last_date"], "--last-date") or latest_price_date()
        if last is None:
            raise CommandError("daily_price is empty")
        try:
            folds = generate_folds(TradingDays.from_db(), last, self._design(options).config)
        except (CalendarError, WalkForwardError) as error:
            raise CommandError(f"--last-date {last}: {error}") from error
        if options["folds"] is None:
            return [fold for fold in folds if not is_holdout(fold)]
        wanted = fold_indices(options["folds"])
        known = {fold.index: fold for fold in folds}
        unknown = sorted(wanted - known.keys())
        held = sorted(index for index in wanted & known.keys() if is_holdout(known[index]))
        problems = [
            *([f"folds {unknown} do not exist"] if unknown else []),
            *([f"holdout folds {held} wait for Phase 4"] if held else []),
        ]
        if problems:
            raise CommandError("; ".join(problems))
        return [known[index] for index in sorted(wanted)]

    def _shuffle(self, folds: list[Fold], seed: int) -> None:
        fold = next((fold for fold in folds if fold.research), folds[0])
        try:
            ic = shuffle_check(fold, settings.FEATURE_PARQUET_DIR, seed)
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            raise CommandError(f"shuffle check on fold {fold.index}: {error}") from error
        self.stdout.write(f"fold {fold.index}: shuffled OOS rank IC {ic:.4f}")
        if math.isnan(ic):
            raise CommandError(f"fold {fold.index} has no OOS day to check; the check cannot pass")
        if abs(ic) >= settings.SHUFFLE_IC_LIMIT:
            raise CommandError(
                f"|IC| {abs(ic):.4f} is not below {settings.SHUFFLE_IC_LIMIT}; "
                "a label column may be among the features"
            )

    def handle(self, *args: Any, **options: Any) -> None:
        folds = self._folds(options)
        if not folds:
            raise CommandError("no research folds before the last date")
        if options["shuffle_check"]:
            self._shuffle(folds, options["seed"])
            return
        scope, run_key = options["scope"], options["run_key"]
        if not scope or not run_key:
            raise CommandError("training needs --scope and --run-key")
        if options["bags"] != 1 and options["model"] != lgbm_runs.MODEL:
            raise CommandError(f"--bags bags LightGBM only, not --model {options['model']}")
        try:
            check_run_key(scope, run_key)
            bag_seeds(options["seed"], options["bags"])
            if options["model"] == lgbm_runs.MODEL:
                lgbm_runs.scope_features(scope, self._design(options).target)
            else:
                scope_features(scope)
        except (ValueError, RunError) as error:
            raise CommandError(str(error)) from error
        if options["folds"] is None:
            self._shuffle(folds, options["seed"])
        try:
            ran = self._train(scope, folds, run_key, options)
        except (HoldoutError, RunError, ValueError, FileNotFoundError, RuntimeError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(f"{run_key}: ran folds {ran}, {len(folds) - len(ran)} cached")

    def _train(
        self, scope: str, folds: list[Fold], run_key: str, options: dict[str, Any]
    ) -> list[int]:
        directory, rerun = settings.FEATURE_PARQUET_DIR, options["rerun"]
        design = self._design(options)
        if options["model"] == lgbm_runs.MODEL:
            return lgbm_runs.run_folds(
                scope,
                folds,
                run_key,
                directory,
                options["seed"],
                rerun,
                options["bags"],
                design.variant,
                design.target,
            )
        return run_folds(scope, folds, run_key, directory, rerun, target=design.target)
