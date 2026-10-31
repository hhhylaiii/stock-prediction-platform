"""Backfill a source dataset stock by stock with a resumable checkpoint (SPEC §5 Phase 1)."""

from argparse import ArgumentParser
from datetime import date, timedelta
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q
from django.utils import timezone

from apps.etl.adapters import build_adapter
from apps.etl.errors import SourceError
from apps.etl.management.options import as_date, required_date
from apps.etl.models import EtlRun
from apps.etl.ratelimit import RetryableError
from apps.market.calendar import CalendarError
from apps.market.models import StockMaster


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


def resume_point(source: str, dataset: str, start: date, end: date) -> str | None:
    """Checkpoint of the latest run with the same arguments, if that run did not finish."""
    latest = (
        EtlRun.objects.filter(source=source, dataset=dataset, start_date=start, end_date=end)
        .order_by("-run_id")
        .first()
    )
    if latest is None or latest.status == EtlRun.Status.SUCCESS:
        raise CommandError(f"no unfinished backfill of {source}/{dataset} {start}..{end} to resume")
    checkpoint = latest.checkpoint.get("last_stock_id")
    return str(checkpoint) if checkpoint is not None else None


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
        end = as_date(options["end"], "--to") or timezone.localdate() - timedelta(days=1)
        if start > end:
            raise CommandError(f"--from {start} is after --to {end}")
        source, dataset = options["source"], options["dataset"]
        try:
            adapter = build_adapter(source)
            adapter.require_dataset(dataset)
            stocks = stocks_alive_in(start, options["stocks"])
            after = resume_point(source, dataset, start, end) if options["resume"] else None
            run = adapter.backfill(dataset, stocks, start, end, resume_after=after)
        except (ValueError, CalendarError, SourceError, RetryableError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        self.stdout.write(
            f"etl_run {run.run_id} {run.status}: {run.rows} rows, last {run.checkpoint}"
        )
