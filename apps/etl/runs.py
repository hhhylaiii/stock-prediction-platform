"""etl_run bookkeeping shared by every ETL entry point (SPEC §5)."""

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from types import FrameType

from django.utils import timezone

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
