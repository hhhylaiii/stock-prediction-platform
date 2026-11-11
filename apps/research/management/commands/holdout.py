"""Run a scope's one holdout on its frozen design, or rehearse it on a research fold (§7)."""

from argparse import ArgumentParser
from datetime import date
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import CalendarError, TradingDays
from apps.research.holdout import HoldoutError
from apps.research.holdout_eval import (
    Header,
    bags_of,
    judge,
    mark_reported,
    render,
    reported_in,
    require_research_caches,
    write_holdout_metrics,
    write_once,
)
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
        "Claim a scope's holdout, fit its holdout folds once on the frozen design and report them "
        "once on the criteria fixed before the run, resuming the latest claim under the same "
        "run_key; --rehearse fits and reports one research fold the same way, without a claim."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--scope", required=True, choices=HoldoutScope.values)
        parser.add_argument("--run-key", required=True)
        parser.add_argument("--output", help="Markdown report, written once; the holdout needs it.")
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
        if output and Path(output).exists():
            raise CommandError(f"{output} exists; a report is written once and never replaced")

    def _research(self, scope: str, last: date) -> tuple[str, list[Fold]]:
        """The frozen research run and its folds, refused before any fit when a cache is gone."""
        design = frozen_design(scope)
        key = str(design.research_run_key)
        research = generate_folds(TradingDays.from_db(), last, design.config)
        require_research_caches(key, research)
        return key, research

    def _report(
        self, header: Header, held: list[Fold], research: list[Fold], output: str | None
    ) -> str:
        design = frozen_design(header.scope)
        verdict, reports = judge(
            header.scope,
            header.run_key,
            held,
            header.research_key,
            research,
            settings.FEATURE_PARQUET_DIR,
            design.target.relative_up,
        )
        if reports and not header.rehearsal:
            stored = write_holdout_metrics(header.run_key, reports)
            self.stdout.write(f"{header.run_key}: {stored} backtest_metric rows")
        bags = {"research": bags_of(header.research_key), "holdout": bags_of(header.run_key)}
        text = render(header, verdict, bags)
        if output:
            write_once(output, text)
            self.stdout.write(f"wrote {output}")
        else:
            self.stdout.write(text)
        return text

    def _rehearse(self, scope: str, run_key: str, options: dict[str, Any]) -> None:
        last = as_date(options["last_date"], "--last-date") or latest_price_date()
        if last is None:
            raise ValueError("daily_price is empty")
        key, research = self._research(scope, last)
        fold = research_fold(scope, options["fold"], last, TradingDays.from_db())
        seed, bags = options["seed"] or 0, options["bags"] or 1
        rehearse(scope, run_key, fold, settings.FEATURE_PARQUET_DIR, seed=seed, bags=bags)
        self.stdout.write(f"{run_key}: rehearsed research fold {fold.index}")
        header = Header(scope, run_key, key, None, last, rehearsal=True)
        self._report(header, [fold], research, options["output"])

    def _holdout(self, scope: str, run_key: str, options: dict[str, Any]) -> None:
        earlier = reported_in(run_key)
        if earlier is not None:
            raise HoldoutError(f"holdout {run_key} was reported in {earlier}; it is reported once")
        last = latest_price_date()
        if last is None:
            raise ValueError("daily_price is empty")
        key, research = self._research(scope, last)
        outcome = run_holdout(
            scope,
            run_key,
            last,
            settings.FEATURE_PARQUET_DIR,
            force=options["force"],
            reason=options["reason"],
        )
        ticket = outcome.ticket
        self.stdout.write(
            f"{run_key}: holdout attempt {ticket.attempt} of {scope} through {ticket.last_date}; "
            f"fitted folds {outcome.ran}, already written {outcome.kept}"
        )
        held = holdout_folds(scope, ticket.last_date, TradingDays.from_db())
        header = Header(scope, run_key, key, ticket.attempt, ticket.last_date, rehearsal=False)
        text = self._report(header, held, research, options["output"])
        mark_reported(run_key, options["output"], text)

    def handle(self, *args: Any, **options: Any) -> None:
        self._check(options)
        scope, run_key = options["scope"], options["run_key"]
        try:
            if options["rehearse"]:
                self._rehearse(scope, run_key, options)
            else:
                self._holdout(scope, run_key, options)
        except ERRORS as error:
            raise CommandError(str(error)) from error
        except RuntimeError as error:
            raise CommandError(f"{error}; run the same command again to resume") from error
