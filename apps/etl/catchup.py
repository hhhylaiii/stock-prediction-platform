"""Fetch the trading days each daily dataset missed, then check those days again (SPEC §5)."""

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import reduce
from itertools import groupby
from operator import or_

from django.db import models
from django.utils import timezone

from apps.etl.adapters import build_adapter
from apps.etl.adapters.dispositions import ATTENTION, DISPOSITION, FULL_DELIVERY
from apps.etl.adapters.exchange_tables import DailyTable
from apps.etl.adapters.tpex_tables import TABLES as TPEX_TABLES
from apps.etl.adapters.twse_tables import TABLES as TWSE_TABLES
from apps.etl.adapters.yahoo import SERIES
from apps.etl.alerts import dispatch
from apps.etl.errors import SourceError, SourceNotReadyError
from apps.etl.models import EtlRun
from apps.etl.quality import run_checks
from apps.etl.ratelimit import RetryableError
from apps.etl.replay import reprocess
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import (
    DailyChip,
    DailyPrice,
    IndexDaily,
    IndexId,
    Market,
    SharesOutstanding,
    StockMaster,
)

logger = logging.getLogger(__name__)
Presence = Callable[[Sequence[date]], set[date]]
NOTHING = "nothing to catch up"


@dataclass(frozen=True)
class Job:
    """One dataset of one source; days count as done by its rows, else by its etl_run spans."""

    source: str
    dataset: str
    present: Presence | None = None
    rebuild_later: bool = False

    @property
    def key(self) -> str:
        return f"{self.source}/{self.dataset}"


@dataclass(frozen=True)
class JobOutcome:
    written: list[date] = field(default_factory=list)
    rebuilt: list[date] = field(default_factory=list)
    not_ready: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CatchupReport:
    written: dict[str, list[str]]
    rebuilt: dict[str, list[str]]
    not_ready: dict[str, str]
    failed: dict[str, str]
    checked: list[str]

    def lines(self) -> list[str]:
        found = [
            *(f"wrote {key}: {', '.join(days)}" for key, days in self.written.items()),
            *(f"rebuilt {key}: {', '.join(days)}" for key, days in self.rebuilt.items()),
            *(f"not ready {key}: {reason}" for key, reason in self.not_ready.items()),
            *(f"failed {key}: {reason}" for key, reason in self.failed.items()),
        ]
        if self.checked:
            found.append(f"checked quality again: {', '.join(self.checked)}")
        return found or [NOTHING]


def rows_present(model: type[models.Model], market: str, columns: Iterable[str] = ()) -> Presence:
    """Days with a row of the market's stocks, with one of the columns filled when given."""
    filled = reduce(
        or_, (models.Q(**{f"{column}__isnull": False}) for column in columns), models.Q()
    )

    def present(days: Sequence[date]) -> set[date]:
        codes = StockMaster.objects.filter(market=market).values("stock_id")
        rows = model._default_manager.filter(filled, date__in=days, stock_id__in=codes)
        return set(rows.values_list("date", flat=True).distinct())

    return present


def index_present(index_id: str) -> Presence:
    def present(days: Sequence[date]) -> set[date]:
        rows = IndexDaily.objects.filter(index_id=index_id, date__in=days)
        return set(rows.values_list("date", flat=True))

    return present


def _table_job(source: str, market: str, dataset: str, table: DailyTable) -> Job:
    """Share counts are stored only on change, so their days count as done by etl_run."""
    if table.model is SharesOutstanding:
        return Job(source, dataset, rebuild_later=True)
    columns = tuple(table.columns) if table.model is DailyChip else ()
    return Job(source, dataset, rows_present(table.model, market, columns))


def _exchange_jobs(
    source: str, market: str, index_id: str, tables: dict[str, DailyTable]
) -> list[Job]:
    return [
        Job(source, "daily_price", rows_present(DailyPrice, market)),
        Job(source, "index_daily", index_present(index_id)),
        *(_table_job(source, market, dataset, table) for dataset, table in tables.items()),
        *(Job(source, dataset) for dataset in (ATTENTION, DISPOSITION, FULL_DELIVERY)),
    ]


JOBS: tuple[Job, ...] = (
    *_exchange_jobs("twse", Market.TWSE.value, IndexId.TAIEX.value, TWSE_TABLES),
    *_exchange_jobs("tpex", Market.TPEX.value, IndexId.TPEX.value, TPEX_TABLES),
    *(Job("yahoo", dataset) for dataset in SERIES),
)


def select_jobs(source: str | None = None, dataset: str | None = None) -> list[Job]:
    return [job for job in JOBS if source in (None, job.source) and dataset in (None, job.dataset)]


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


def missing_days(job: Job, days: Sequence[date]) -> list[date]:
    done = job.present(days) if job.present else covered_days(job.source, job.dataset, days)
    return [day for day in days if day not in done]


def catch_up(
    since: date, until: date, calendar: TradingDays, jobs: Sequence[Job] = JOBS
) -> CatchupReport:
    """Each job day by day in date order; a job stops at its first failure, others go on."""
    today = timezone.localdate()
    if until > today:
        raise CalendarError(f"{until} is after today ({today}); nothing has been published")
    days = calendar.between(since, until)
    outcomes = {job.key: _catch_up_job(job, days, calendar) for job in jobs}
    written = sorted({day for outcome in outcomes.values() for day in outcome.written})
    for day in written:
        dispatch(run_checks(day, calendar))
    return CatchupReport(
        written=_dated(outcomes, "written"),
        rebuilt=_dated(outcomes, "rebuilt"),
        not_ready={k: v for outcome in outcomes.values() for k, v in outcome.not_ready.items()},
        failed={k: v for outcome in outcomes.values() for k, v in outcome.failed.items()},
        checked=[day.isoformat() for day in written],
    )


def _dated(outcomes: dict[str, JobOutcome], name: str) -> dict[str, list[str]]:
    return {
        key: [day.isoformat() for day in getattr(outcome, name)]
        for key, outcome in outcomes.items()
        if getattr(outcome, name)
    }


def _catch_up_job(job: Job, days: Sequence[date], calendar: TradingDays) -> JobOutcome:
    adapter = build_adapter(job.source, calendar)
    written: list[date] = []
    not_ready: dict[str, str] = {}
    failed: dict[str, str] = {}
    for day in missing_days(job, days):
        try:
            adapter.run(job.dataset, day, day)
        except SourceNotReadyError as exc:
            not_ready[f"{job.key} {day}"] = str(exc)
            continue
        except (SourceError, RetryableError) as exc:
            failed[f"{job.key} {day}"] = f"{type(exc).__name__}: {exc}"
            logger.warning("catchup %s %s failed; later days wait: %s", job.key, day, exc)
            break
        written.append(day)
    rebuilt: list[date] = []
    if job.rebuild_later and written:
        rebuilt, rebuild_failed = _rebuild_after(job, written[0], calendar)
        failed = {**failed, **rebuild_failed}
    return JobOutcome(written, rebuilt, not_ready, failed)


def _rebuild_after(
    job: Job, first: date, calendar: TradingDays
) -> tuple[list[date], dict[str, str]]:
    """Replay every later fetched day in order, so values kept only on change stay right."""
    end = min(calendar.end, timezone.localdate())
    later = [day for day in calendar.between(first, end) if day > first]
    fetched = covered_days(job.source, job.dataset, later)
    rebuilt: list[date] = []
    for is_fetched, group in groupby(later, key=lambda day: day in fetched):
        span = list(group)
        if not is_fetched:
            continue
        try:
            reprocess(job.source, job.dataset, span[0], span[-1], calendar)
        except (SourceError, RetryableError) as exc:
            return rebuilt, {f"{job.key} rebuild {span[0]}": f"{type(exc).__name__}: {exc}"}
        rebuilt.extend(span)
    return rebuilt, {}
