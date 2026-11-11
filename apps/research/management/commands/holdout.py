"""Run a scope's one holdout on its frozen design, or rehearse it on a research fold (§7)."""

from argparse import ArgumentParser
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import CalendarError, TradingDays
from apps.research.holdout import HoldoutError
from apps.research.holdout_runs import rehearse, research_fold, run_holdout
from apps.research.labels import latest_price_date
from apps.research.models import HoldoutScope
from apps.research.production import BAGS
from apps.research.runs import RunError
from apps.research.walkforward import WalkForwardError


class Command(BaseCommand):
    help = (
        "Claim a scope's holdout and fit its holdout folds once on the frozen design, resuming "
        "the latest claim under the same run_key; --rehearse fits one research fold the same "
        "way under a rehearsal key, without a claim."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--scope", required=True, choices=HoldoutScope.values)
        parser.add_argument("--run-key", required=True)
        parser.add_argument(
            "--last-date", help="Last price day of a new claim; today's by default."
        )
        parser.add_argument(
            "--bags", type=int, default=BAGS, help="LightGBM seeds bagged per model."
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--force", action="store_true", help="A new claim after an earlier one."
        )
        parser.add_argument("--reason", default="", help="Why a forced claim reruns the holdout.")
        parser.add_argument(
            "--rehearse",
            action="store_true",
            help="Fit one research fold on the holdout's path under a <scope>-rehearsal key.",
        )
        parser.add_argument(
            "--fold", type=int, help="Research fold of the rehearsal; the last one by default."
        )

    def _check(self, options: dict[str, Any]) -> None:
        if options["rehearse"] and (options["force"] or options["reason"]):
            raise CommandError("--force and --reason claim a holdout; a rehearsal claims none")
        if not options["rehearse"] and options["fold"] is not None:
            raise CommandError("--fold picks the rehearsal's research fold; add --rehearse")
        if options["bags"] < 1:
            raise CommandError("--bags must be at least 1")

    def handle(self, *args: Any, **options: Any) -> None:
        self._check(options)
        scope, run_key = options["scope"], options["run_key"]
        directory = settings.FEATURE_PARQUET_DIR
        try:
            last = as_date(options["last_date"], "--last-date") or latest_price_date()
            if last is None:
                raise ValueError("daily_price is empty")
            if options["rehearse"]:
                fold = research_fold(scope, options["fold"], last, TradingDays.from_db())
                rehearse(
                    scope, run_key, fold, directory, seed=options["seed"], bags=options["bags"]
                )
                self.stdout.write(f"{run_key}: rehearsed research fold {fold.index}")
                return
            outcome = run_holdout(
                scope,
                run_key,
                last,
                directory,
                seed=options["seed"],
                bags=options["bags"],
                force=options["force"],
                reason=options["reason"],
            )
        except (
            ValueError,
            LookupError,
            OSError,
            CalendarError,
            WalkForwardError,
            HoldoutError,
            RunError,
        ) as error:
            raise CommandError(str(error)) from error
        ticket = outcome.ticket
        self.stdout.write(
            f"{run_key}: holdout attempt {ticket.attempt} of {scope} through {ticket.last_date}; "
            f"fitted folds {outcome.ran}, already written {outcome.kept}"
        )
