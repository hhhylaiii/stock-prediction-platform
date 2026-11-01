"""Backfill a source dataset stock by stock with a resumable checkpoint (SPEC §5 Phase 1)."""

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q
from django.utils import timezone

from apps.etl.adapters import build_adapter
from apps.etl.adapters.base import stock_list_digest
from apps.etl.errors import RejectedStocksError, SourceError
from apps.etl.management.options import as_date, required_date
from apps.etl.models import EtlRun
from apps.etl.ratelimit import RetryableError
from apps.etl.runs import failing_on_termination
from apps.market.calendar import CalendarError
from apps.market.models import StockMaster

RANGE_OPTIONS = ("start", "end", "stocks", "resume")


def stocks_alive_in(start: date, chosen: list[str] | None) -> list[str]:
    """Listed stocks plus those delisted on or after start, optionally narrowed to chosen ones."""
    alive = StockMaster.objects.filter(Q(delist_date__isnull=True) | Q(delist_date__gte=start))
    codes = sorted(alive.values_list("stock_id", flat=True))
    if not StockMaster.objects.exists():
        raise CommandError("stock_master is empty; run manage.py load_stock_master first")
    if not chosen:
        return codes
    unknown = sorted(set(chosen) - set(StockMaster.objects.values_list("stock_id", flat=True)))
    if unknown:
        raise CommandError(f"not in stock_master: {', '.join(unknown)}")
    return sorted(set(chosen))


@dataclass(frozen=True)
class ResumePoint:
    run_id: int
    end: date
    after: str | None
    stocks: int | None
    digest: str | None
    rejected: dict[str, str]


@dataclass(frozen=True)
class Plan:
    start: date
    end: date
    stocks: list[str]
    selection: str | list[str]
    after: str | None = None
    rejected: dict[str, str] | None = None


def resume_point(
    source: str, dataset: str, start: date, end: date | None, selection: str | list[str]
) -> ResumePoint:
    """The latest backfill with the same start and selection, if it failed part way."""
    runs = EtlRun.objects.filter(
        source=source,
        dataset=dataset,
        start_date=start,
        checkpoint__kind="backfill",
        checkpoint__selection=selection,
    )
    if end is not None:
        runs = runs.filter(end_date=end)
    latest = runs.order_by("-run_id").first()
    span = f"{start}..{end}" if end is not None else f"from {start}"
    if latest is None or latest.status == EtlRun.Status.SUCCESS:
        raise CommandError(f"no unfinished backfill of {source}/{dataset} {span} to resume")
    if latest.status == EtlRun.Status.RUNNING:
        raise CommandError(
            f"etl_run {latest.run_id} is still marked running; stop that process first, "
            "or set its status to failed if it died"
        )
    checkpoint = latest.checkpoint
    if checkpoint.get("finished"):
        raise CommandError(
            f"etl_run {latest.run_id} reached its last stock with rejected stocks; "
            f"rerun them with --retry-rejected {latest.run_id}"
        )
    after = checkpoint.get("last_stock_id")
    return ResumePoint(
        run_id=latest.run_id,
        end=latest.end_date,
        after=str(after) if after is not None else None,
        stocks=checkpoint.get("stocks"),
        digest=checkpoint.get("digest"),
        rejected=dict(checkpoint.get("rejected", {})),
    )


def require_same_stocks(point: ResumePoint, stocks: list[str]) -> None:
    """A different stock list would silently skip codes added before the checkpoint."""
    if point.digest != stock_list_digest(stocks):
        raise CommandError(
            f"stock list changed since etl_run {point.run_id} started, from {point.stocks} "
            f"to {len(stocks)} stocks; rerun without --resume, or back up the rest with --stock"
        )


def retry_plan(source: str, dataset: str, run_id: int) -> Plan:
    """The rejected stocks of an earlier backfill, over that run's range."""
    earlier = EtlRun.objects.filter(run_id=run_id, source=source, dataset=dataset).first()
    if earlier is None:
        raise CommandError(f"no etl_run {run_id} of {source}/{dataset}")
    if earlier.status == EtlRun.Status.FAILED and not earlier.checkpoint.get("finished"):
        raise CommandError(f"etl_run {run_id} stopped before its last stock; --resume it first")
    if earlier.status == EtlRun.Status.RUNNING:
        raise CommandError(f"etl_run {run_id} is still marked running")
    rejected = sorted(earlier.checkpoint.get("rejected", {}))
    if not rejected:
        raise CommandError(f"etl_run {run_id} has no rejected stocks")
    return Plan(earlier.start_date, earlier.end_date, rejected, rejected)


def range_plan(source: str, dataset: str, options: dict[str, Any]) -> Plan:
    start = required_date(options["start"], "--from")
    explicit_end = as_date(options["end"], "--to")
    stocks = stocks_alive_in(start, options["stocks"])
    selection: str | list[str] = stocks if options["stocks"] else "all"
    end = explicit_end or timezone.localdate() - timedelta(days=1)
    if not options["resume"]:
        return Plan(start, end, stocks, selection)
    point = resume_point(source, dataset, start, explicit_end, selection)
    require_same_stocks(point, stocks)
    return Plan(start, point.end, stocks, selection, point.after, point.rejected)


class Command(BaseCommand):
    help = "Backfill a dataset for every stock in stock_master, resumable after interruptions."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--source", required=True)
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--from", dest="start")
        parser.add_argument("--to", dest="end")
        parser.add_argument("--stock", action="append", dest="stocks")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument(
            "--retry-rejected",
            type=int,
            metavar="RUN_ID",
            help="rerun only the stocks that run rejected, over its range",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        source, dataset = options["source"], options["dataset"]
        retry = options["retry_rejected"]
        if retry is not None and any(options[key] for key in RANGE_OPTIONS):
            raise CommandError("--retry-rejected takes its range and stocks from that run")
        try:
            adapter = build_adapter(source)
            adapter.require_dataset(dataset)
            plan = (
                retry_plan(source, dataset, retry)
                if retry is not None
                else range_plan(source, dataset, options)
            )
            if plan.start > plan.end:
                raise CommandError(f"--from {plan.start} is after --to {plan.end}")
            with failing_on_termination():
                run = adapter.backfill(
                    dataset,
                    plan.stocks,
                    plan.start,
                    plan.end,
                    resume_after=plan.after,
                    selection=plan.selection,
                    rejected=plan.rejected,
                    guard_streaks=retry is None,
                )
        except RejectedStocksError as exc:
            raise CommandError(str(exc)) from exc
        except (ValueError, CalendarError, SourceError, RetryableError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        self.stdout.write(
            f"etl_run {run.run_id} {run.status}: {run.rows} rows, last {run.checkpoint}"
        )
