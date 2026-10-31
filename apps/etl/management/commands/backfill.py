"""Backfill a source dataset stock by stock with a resumable checkpoint (SPEC §5 Phase 1)."""

import signal
from argparse import ArgumentParser
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from types import FrameType
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q
from django.utils import timezone

from apps.etl.adapters import build_adapter
from apps.etl.adapters.base import stock_list_digest
from apps.etl.errors import SourceError
from apps.etl.management.options import as_date, required_date
from apps.etl.models import EtlRun
from apps.etl.ratelimit import RetryableError
from apps.market.calendar import CalendarError
from apps.market.models import StockMaster

STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


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
    after = checkpoint.get("last_stock_id")
    return ResumePoint(
        run_id=latest.run_id,
        end=latest.end_date,
        after=str(after) if after is not None else None,
        stocks=checkpoint.get("stocks"),
        digest=checkpoint.get("digest"),
    )


def require_same_stocks(point: ResumePoint, stocks: list[str]) -> None:
    """A different stock list would silently skip codes added before the checkpoint."""
    if point.digest != stock_list_digest(stocks):
        raise CommandError(
            f"stock list changed since etl_run {point.run_id} started, from {point.stocks} "
            f"to {len(stocks)} stocks; rerun without --resume, or back up the rest with --stock"
        )


def _terminate(signum: int, frame: FrameType | None) -> None:
    """SIGTERM and SIGHUP end the run as failed, keeping its checkpoint for --resume."""
    raise SystemExit(128 + signum)


@contextmanager
def failing_on_termination() -> Iterator[None]:
    previous = {signum: signal.signal(signum, _terminate) for signum in STOP_SIGNALS}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class Command(BaseCommand):
    help = "Backfill a dataset for every stock in stock_master, resumable after interruptions."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--source", required=True)
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--from", dest="start", required=True)
        parser.add_argument("--to", dest="end")
        parser.add_argument("--stock", action="append", dest="stocks")
        parser.add_argument("--resume", action="store_true")

    def handle(self, *args: Any, **options: Any) -> None:
        start = required_date(options["start"], "--from")
        explicit_end = as_date(options["end"], "--to")
        source, dataset = options["source"], options["dataset"]
        try:
            adapter = build_adapter(source)
            adapter.require_dataset(dataset)
            stocks = stocks_alive_in(start, options["stocks"])
            selection: str | list[str] = stocks if options["stocks"] else "all"
            after = None
            end = explicit_end or timezone.localdate() - timedelta(days=1)
            if options["resume"]:
                point = resume_point(source, dataset, start, explicit_end, selection)
                require_same_stocks(point, stocks)
                end, after = point.end, point.after
            if start > end:
                raise CommandError(f"--from {start} is after --to {end}")
            with failing_on_termination():
                run = adapter.backfill(
                    dataset, stocks, start, end, resume_after=after, selection=selection
                )
        except (ValueError, CalendarError, SourceError, RetryableError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        self.stdout.write(
            f"etl_run {run.run_id} {run.status}: {run.rows} rows, last {run.checkpoint}"
        )
