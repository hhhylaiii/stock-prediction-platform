"""Run a scope's one holdout on its frozen design, or rehearse it on a research fold (§7)."""

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import CalendarError, TradingDays
from apps.research.ensemble import write_report
from apps.research.holdout import HoldoutError
from apps.research.holdout_eval import Header, bags_of, judge, render, write_holdout_metrics
from apps.research.holdout_runs import (
    frozen_design,
    holdout_folds,
    rehearse,
    research_fold,
    run_holdout,
)
from apps.research.labels import latest_price_date
from apps.research.models import HoldoutScope
from apps.research.prediction_cache import FoldMismatchError, MissingFoldsError
from apps.research.production import BAGS
from apps.research.runs import RunError
from apps.research.walkforward import Fold, WalkForwardError, generate_folds

ERRORS = (
    ValueError,
    LookupError,
    OSError,
    CalendarError,
    WalkForwardError,
    HoldoutError,
    RunError,
    FoldMismatchError,
    MissingFoldsError,
)


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
            "--force", action="store_true", help="A new claim after an earlier one."
        )
        parser.add_argument("--reason", default="", help="Why a forced claim reruns the holdout.")
        parser.add_argument(
            "--rehearse",
            action="store_true",
            help="Fit one research fold on the holdout's path under a <scope>-rehearsal key.",
        )
        parser.add_argument(
            "--fold", type=int, help="Rehearsal only: its research fold, the last one by default."
        )
        parser.add_argument(
            "--last-date", help="Rehearsal only: last price day; the holdout claims the latest."
        )
        parser.add_argument(
            "--bags", type=int, help=f"Rehearsal only, 1 by default; the holdout bags {BAGS}."
        )
        parser.add_argument("--seed", type=int, help="Rehearsal only, 0 by default.")
        parser.add_argument(
            "--output",
            help="Markdown report to write once; the holdout needs it, a rehearsal prints without.",
        )

    def _check(self, options: dict[str, Any]) -> None:
        if options["rehearse"] and (options["force"] or options["reason"]):
            raise CommandError("--force and --reason claim a holdout; a rehearsal claims none")
        fixed = [
            name for name in ("fold", "last_date", "bags", "seed") if options[name] is not None
        ]
        if not options["rehearse"] and fixed:
            names = ", ".join(f"--{name.replace('_', '-')}" for name in fixed)
            raise CommandError(f"{names} apply to a rehearsal; the holdout runs as frozen")
        if options["bags"] is not None and options["bags"] < 1:
            raise CommandError("--bags must be at least 1")
        output = options["output"]
        if not options["rehearse"] and not output:
            raise CommandError("the holdout writes its report once; give --output")
        if output and not options["rehearse"] and Path(output).exists():
            raise CommandError(f"{output} exists; the holdout is reported once")

    def _report(
        self, scope: str, run_key: str, held: list[Fold], header: Header, options: dict[str, Any]
    ) -> None:
        directory = settings.FEATURE_PARQUET_DIR
        design = frozen_design(scope)
        key = str(design.research_run_key)
        research = generate_folds(TradingDays.from_db(), header.last_date, design.config)
        verdict, reports = judge(
            scope, run_key, held, key, research, directory, design.target.relative_up
        )
        bags = {"research": bags_of(key), "holdout": bags_of(run_key)}
        text = render(replace(header, research_key=key), verdict, bags)
        written = write_report(text, options["output"])
        if not header.rehearsal:
            stored = write_holdout_metrics(run_key, reports)
            self.stdout.write(f"{run_key}: {stored} backtest_metric rows")
        self.stdout.write(text if written is None else f"wrote {written}")

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
                seed, bags = options["seed"] or 0, options["bags"] or 1
                rehearse(scope, run_key, fold, directory, seed=seed, bags=bags)
                self.stdout.write(f"{run_key}: rehearsed research fold {fold.index}")
                header = Header(scope, run_key, "", None, last, rehearsal=True)
                self._report(scope, run_key, [fold], header, options)
                return
            outcome = run_holdout(
                scope,
                run_key,
                last,
                directory,
                force=options["force"],
                reason=options["reason"],
            )
            ticket = outcome.ticket
            self.stdout.write(
                f"{run_key}: holdout attempt {ticket.attempt} of {scope} through "
                f"{ticket.last_date}; fitted folds {outcome.ran}, already written {outcome.kept}"
            )
            held = holdout_folds(scope, ticket.last_date, TradingDays.from_db())
            header = Header(scope, run_key, "", ticket.attempt, ticket.last_date, rehearsal=False)
            self._report(scope, run_key, held, header, options)
        except ERRORS as error:
            raise CommandError(str(error)) from error
        except RuntimeError as error:
            raise CommandError(f"{error}; run the same command again to resume") from error
