"""Fit a target family's baselines fold by fold, or run the label shuffle check (SPEC §7, §8)."""

from argparse import ArgumentParser
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import TradingDays
from apps.research.baseline_runs import SCOPES, run_folds, shuffle_check
from apps.research.holdout import HoldoutError, is_holdout
from apps.research.labels import latest_price_date
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

DEFAULT_CONFIG = WalkForwardConfig()


def fold_indices(text: str) -> set[int]:
    """Indices from a list like 0-3,7."""
    chosen: set[int] = set()
    for part in text.split(","):
        low, _, high = part.strip().partition("-")
        try:
            chosen.update(range(int(low), int(high or low) + 1))
        except ValueError:
            raise CommandError(f"--folds takes indices like 0-3,7, got {text!r}") from None
    return chosen


class Command(BaseCommand):
    help = "Fit one scope's baselines on research folds, caching predictions; or --shuffle-check."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--scope", choices=list(SCOPES))
        parser.add_argument("--run-key")
        parser.add_argument("--folds")
        parser.add_argument("--last-date")
        parser.add_argument("--rerun", action="store_true")
        parser.add_argument("--shuffle-check", action="store_true")
        parser.add_argument("--seed", type=int, default=0)

    def _folds(self, options: dict[str, Any]) -> list[Fold]:
        last = as_date(options["last_date"], "--last-date") or latest_price_date()
        if last is None:
            raise CommandError("daily_price is empty")
        folds = generate_folds(TradingDays.from_db(), last, DEFAULT_CONFIG)
        if options["folds"] is None:
            return [fold for fold in folds if not is_holdout(fold)]
        wanted = fold_indices(options["folds"])
        known = {fold.index: fold for fold in folds}
        unknown = sorted(wanted - known.keys())
        held = sorted(index for index in wanted & known.keys() if is_holdout(known[index]))
        if unknown or held:
            raise CommandError(
                f"folds {unknown} do not exist; holdout folds {held} wait for Phase 4"
            )
        return [known[index] for index in sorted(wanted)]

    def _shuffle(self, folds: list[Fold], seed: int) -> None:
        fold = next((fold for fold in folds if fold.research), folds[0])
        ic = shuffle_check(fold, settings.FEATURE_PARQUET_DIR, seed)
        self.stdout.write(f"fold {fold.index}: shuffled OOS rank IC {ic:.4f}")
        if abs(ic) >= settings.SHUFFLE_IC_LIMIT:
            raise CommandError(
                f"|IC| {abs(ic):.4f} is not below {settings.SHUFFLE_IC_LIMIT}; "
                "look for a leak in the features or the split"
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
        try:
            ran = run_folds(scope, folds, run_key, settings.FEATURE_PARQUET_DIR, options["rerun"])
        except (HoldoutError, ValueError, FileNotFoundError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(f"{run_key}: ran folds {ran}, {len(folds) - len(ran)} cached")
