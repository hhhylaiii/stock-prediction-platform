"""Fetch the trading days each daily dataset missed, then check those days again (SPEC §5)."""

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import reduce
from operator import or_

from django.db import models
from django.utils import timezone

from apps.etl.adapters import build_adapter
from apps.etl.adapters.corporate_actions import DIVIDEND_EVENT
from apps.etl.adapters.dispositions import ATTENTION, DISPOSITION, FULL_DELIVERY
from apps.etl.adapters.exchange_tables import DailyTable
from apps.etl.adapters.tpex import QUOTED_TYPE as TPEX_QUOTED_TYPE
from apps.etl.adapters.tpex_tables import TABLES as TPEX_TABLES
from apps.etl.adapters.twse_tables import TABLES as TWSE_TABLES
from apps.etl.adapters.yahoo import SERIES
from apps.etl.alerts import dispatch
from apps.etl.errors import SourceError, SourceNotReadyError
from apps.etl.models import DataQualityLog, RawTpex, RawTwse
from apps.etl.quality import CHECKS, run_checks
from apps.etl.ratelimit import RetryableError
from apps.etl.replay import rebuild_after
from apps.etl.runs import covered_days
from apps.etl.task_alerts import report_swallowed
from apps.market.calendar import CalendarError, OutsideCalendarError, TradingDays
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
REDONE = {"day_trade": 2}


@dataclass(frozen=True)
class Job:
    """One dataset of one source; days count as done by its rows, else by its etl_run spans.

    redone: sessions before each day that its nightly run fetches again, as the source revises them.
    """

    source: str
    dataset: str
    present: Presence | None = None
    rebuild_later: bool = False
    redone: int = 0

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


def quotes_present(market: str) -> Presence:
    """Days with the market's rows and the exchange's own quotes stored, which chip rows read."""
    rows = rows_present(DailyPrice, market)
    raw_model = RawTwse if market == Market.TWSE.value else RawTpex
    extra = {} if market == Market.TWSE.value else {"request_params__type": TPEX_QUOTED_TYPE}

    def present(days: Sequence[date]) -> set[date]:
        asked = {f"{day:%Y%m%d}": day for day in days}
        stored = raw_model.objects.filter(
            dataset="daily_price", payload__date__in=list(asked), **extra
        ).values_list("payload__date", flat=True)
        return rows(days) & {asked[key] for key in stored if key in asked}

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
    present = rows_present(table.model, market, columns)
    return Job(source, dataset, present, redone=REDONE.get(dataset, 0))


def _exchange_jobs(
    source: str, market: str, index_id: str, tables: dict[str, DailyTable]
) -> list[Job]:
    return [
        Job(source, "daily_price", quotes_present(market)),
        Job(source, "index_daily", index_present(index_id)),
        *(_table_job(source, market, dataset, table) for dataset, table in tables.items()),
        *(
            Job(source, dataset)
            for dataset in (ATTENTION, DISPOSITION, FULL_DELIVERY, DIVIDEND_EVENT)
        ),
    ]


JOBS: tuple[Job, ...] = (
    *_exchange_jobs("twse", Market.TWSE.value, IndexId.TAIEX.value, TWSE_TABLES),
    Job("twse", "return_index", index_present(IndexId.TAIEX_TR.value)),
    *_exchange_jobs("tpex", Market.TPEX.value, IndexId.TPEX.value, TPEX_TABLES),
    *(Job("yahoo", dataset) for dataset in SERIES),
)


def select_jobs(source: str | None = None, dataset: str | None = None) -> list[Job]:
    return [job for job in JOBS if source in (None, job.source) and dataset in (None, job.dataset)]


def plan_catch_up(
    since: date, until: date, calendar: TradingDays, jobs: Sequence[Job] = JOBS
) -> dict[str, list[str]]:
    """The days each job would fetch, without fetching anything."""
    days = calendar.between(since, until)
    planned = {job.key: days_to_fetch(job, days, calendar) for job in jobs}
    return {key: [day.isoformat() for day in found] for key, found in planned.items() if found}


def missing_days(job: Job, days: Sequence[date]) -> list[date]:
    done = job.present(days) if job.present else covered_days(job.source, job.dataset, days)
    return [day for day in days if day not in done]


def days_to_fetch(job: Job, days: Sequence[date], calendar: TradingDays) -> list[date]:
    """Missing days, with the sessions their missed nightly runs would have fetched again."""
    missing = missing_days(job, days)
    again = (_shifted(calendar, day, -back) for day in missing for back in range(1, job.redone + 1))
    return sorted({*missing, *(day for day in again if day is not None)})


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
    checked = _days_to_check(written, calendar)
    for day in checked:
        dispatch(run_checks(day, calendar))
    return CatchupReport(
        written=_dated(outcomes, "written"),
        rebuilt=_dated(outcomes, "rebuilt"),
        not_ready={k: v for outcome in outcomes.values() for k, v in outcome.not_ready.items()},
        failed={k: v for outcome in outcomes.values() for k, v in outcome.failed.items()},
        checked=[day.isoformat() for day in checked],
    )


def _days_to_check(written: Sequence[date], calendar: TradingDays) -> list[date]:
    """Written days, plus each next session already checked against the gap before it."""
    sessions = (_shifted(calendar, day, 1) for day in written)
    following = {day for day in sessions if day is not None} - set(written)
    checked = DataQualityLog.objects.filter(date__in=following, check_name__in=CHECKS).values_list(
        "date", flat=True
    )
    return sorted({*written, *checked})


def _shifted(calendar: TradingDays, day: date, sessions: int) -> date | None:
    try:
        return calendar.shift(day, sessions)
    except OutsideCalendarError:
        return None


def _dated(outcomes: dict[str, JobOutcome], name: str) -> dict[str, list[str]]:
    return {
        key: [day.isoformat() for day in getattr(outcome, name)]
        for key, outcome in outcomes.items()
        if getattr(outcome, name)
    }


def _catch_up_job(job: Job, days: Sequence[date], calendar: TradingDays) -> JobOutcome:
    try:
        adapter = build_adapter(job.source, calendar)
        missing = days_to_fetch(job, days, calendar)
    except Exception as exc:
        return JobOutcome([], [], {}, {f"{job.key} setup": _failure(job, "setup", exc)})
    written: list[date] = []
    not_ready: dict[str, str] = {}
    failed: dict[str, str] = {}
    for day in missing:
        try:
            adapter.run(job.dataset, day, day)
        except SourceNotReadyError as exc:
            not_ready[f"{job.key} {day}"] = str(exc)
            continue
        except Exception as exc:
            failed[f"{job.key} {day}"] = _failure(job, f"{day}; later days wait", exc)
            break
        written.append(day)
    rebuilt: list[date] = []
    if job.rebuild_later and written:
        try:
            rebuilt = rebuild_after(job.source, job.dataset, written[0], calendar)
        except Exception as exc:
            failed[f"{job.key} rebuild after {written[0]}"] = _failure(
                job, f"rebuild after {written[0]}", exc
            )
    return JobOutcome(written, rebuilt, not_ready, failed)


def _failure(job: Job, step: str, exc: Exception) -> str:
    """Log a failed step; one that is not a source answering badly gets a traceback and an alert."""
    expected = isinstance(exc, (SourceError, RetryableError))
    logger.error("catchup failed %s %s: %s", job.key, step, exc, exc_info=not expected)
    if not expected:
        report_swallowed(exc, f"catchup {job.key} {step}")
    return f"{type(exc).__name__}: {exc}"
