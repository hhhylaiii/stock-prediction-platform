"""etl_run bookkeeping shared by every ETL entry point (SPEC §5)."""

import signal
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date
from types import FrameType

from django.conf import settings
from django.utils import timezone

from apps.etl.errors import RejectedStocksError
from apps.etl.models import EtlRun

ERROR_TEXT_LIMIT = 4000
STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


@contextmanager
def tracked_run(source: str, dataset: str, start: date, end: date) -> Iterator[EtlRun]:
    """One etl_run row that ends as success, or failed with the error, even on Ctrl-C."""
    run = EtlRun.objects.create(source=source, dataset=dataset, start_date=start, end_date=end)
    try:
        yield run
    except BaseException as exc:
        run.status = EtlRun.Status.FAILED
        run.error = f"{type(exc).__name__}: {exc}"[:ERROR_TEXT_LIMIT]
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "error", "rows", "checkpoint", "end_date", "finished_at"])
        raise
    run.status = EtlRun.Status.SUCCESS
    run.finished_at = timezone.now()
    run.save(update_fields=["status", "rows", "checkpoint", "end_date", "finished_at"])


def _terminate(signum: int, frame: FrameType | None) -> None:
    """SIGTERM and SIGHUP end the run as failed, keeping its checkpoint."""
    raise SystemExit(128 + signum)


@contextmanager
def failing_on_termination() -> Iterator[None]:
    previous = {signum: signal.signal(signum, _terminate) for signum in STOP_SIGNALS}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def stop_on_streak(streak: Sequence[str], rejected: Mapping[str, str]) -> None:
    """Many rejections in a row look like a blocked source, so stop asking."""
    if streak and len(streak) >= settings.ETL_MAX_CONSECUTIVE_REJECTIONS:
        last = streak[-1]
        raise RejectedStocksError(
            f"{len(streak)} stocks rejected in a row; last: {last}: {rejected[last]}"
        )


def fail_on_rejections(rejected: Mapping[str, str]) -> None:
    """Fail the run listing every rejected stock, with the reason of the first one."""
    if rejected:
        codes = sorted(rejected)
        noun = "stock" if len(codes) == 1 else "stocks"
        raise RejectedStocksError(
            f"{len(codes)} {noun} rejected: {', '.join(codes)}; {codes[0]}: {rejected[codes[0]]}"
        )


def covered_days(source: str, dataset: str, days: Sequence[date]) -> set[date]:
    """Days inside a successful run, or inside the checkpointed part of an unfinished one."""
    if not days:
        return set()
    runs = EtlRun.objects.filter(
        source=source, dataset=dataset, start_date__lte=max(days), end_date__gte=min(days)
    ).values_list("start_date", "end_date", "status", "checkpoint")
    spans = [
        (start, _done_until(end, status, checkpoint)) for start, end, status, checkpoint in runs
    ]
    return {day for day in days for start, until in spans if until and start <= day <= until}


def _done_until(end: date, status: str, checkpoint: dict[str, object]) -> date | None:
    if status == EtlRun.Status.SUCCESS:
        return end
    last = checkpoint.get("last_date")
    return date.fromisoformat(last) if isinstance(last, str) else None
