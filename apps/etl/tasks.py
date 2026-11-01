"""Celery tasks behind the daily ETL schedule (SPEC §5 每日增量, 重試, 失敗處理)."""

from __future__ import annotations

import logging
from datetime import date, time, timedelta
from typing import Any, NamedTuple

from celery import Task, shared_task
from django.conf import settings
from django.core.management import call_command
from django.db import transaction
from django.utils import timezone

from apps.etl.adapters import build_adapter
from apps.etl.alerts import dispatch
from apps.etl.catchup import catch_up
from apps.etl.errors import SourceError, SourceNotReadyError
from apps.etl.master_refresh import refresh_stock_master
from apps.etl.models import DataQualityLog
from apps.etl.quality import CHECKS, run_checks, should_run_prediction
from apps.etl.ratelimit import RetryableError
from apps.etl.runs import covered_days
from apps.market.calendar import TradingDays
from apps.market.models import IndexDaily, IndexId

logger = logging.getLogger(__name__)
US_SERIES = {
    "sox": IndexId.SOX.value,
    "tsm_adr": IndexId.TSM_ADR.value,
    "usdtwd": IndexId.USDTWD.value,
}
US_CLOSE_CHECK = "us_close"
ETL_RUNS_CHECK = "etl_runs"
US_LAG = {IndexId.USDTWD.value: 1}
SATURDAY = 5
BATCH_SLOTS = {
    "prices": "PRICES",
    "afternoon": "AFTERNOON",
    "qfii": "QFII",
    "night": "NIGHT",
    "notices": "NOTICES_AGAIN",
}
EXCHANGES = ("twse", "tpex")
LAST_THREE = (0, 1, 2)


class Fetch(NamedTuple):
    """One dataset of one source, run for today and the given sessions before it."""

    source: str
    dataset: str
    sessions_back: tuple[int, ...] = (0,)


def _both(*datasets: str, sessions_back: tuple[int, ...] = (0,)) -> list[Fetch]:
    return [Fetch(source, name, sessions_back) for name in datasets for source in EXCHANGES]


BATCHES: dict[str, list[Fetch]] = {
    "prices": _both("daily_price", "index_daily"),
    "afternoon": [
        *_both("institutional", "daily_valuation", "attention", "disposition", "full_delivery"),
        Fetch("twse", "foreign_holding"),
        Fetch("twse", "shares_outstanding"),
    ],
    "qfii": [Fetch("tpex", "foreign_holding"), Fetch("tpex", "shares_outstanding")],
    "night": [*_both("margin", "sbl"), *_both("day_trade", sessions_back=LAST_THREE)],
    "notices": _both("attention", "disposition"),
}


def _session_today(slot: str | None = None) -> tuple[TradingDays, date] | None:
    """Today when it is a session and its slot has come; a run fired late on waking is skipped."""
    now = timezone.localtime()
    if slot is not None and now.time() < time.fromisoformat(settings.ETL_AT[slot]):
        return None
    calendar = TradingDays.from_db()
    today = now.date()
    return (calendar, today) if calendar.is_trading_day(today) else None


def _check(day: date, calendar: TradingDays) -> list[DataQualityLog]:
    results = run_checks(day, calendar)
    dispatch(results)
    return results


def _recheck_if_blocked(day: date) -> None:
    """A dataset landing after the day's checks failed re-runs them, so the gate can open."""
    checks = DataQualityLog.objects.filter(date=day, check_name__in=CHECKS)
    if checks.filter(severity=DataQualityLog.Severity.ERROR).exists():
        _check(day, TradingDays.from_db())


def _not_ready(task: Task[Any, Any], what: str, exc: SourceNotReadyError) -> str:
    """Retry later while retries last; then log that the source never published."""
    if task.request.retries >= settings.ETL_NOT_READY_RETRIES:
        logger.error("etl gave up %s after %d retries: %s", what, task.request.retries, exc)
        return "not ready"
    raise task.retry(exc=exc, countdown=settings.ETL_NOT_READY_COUNTDOWN)


def _failed(what: str, exc: Exception) -> str:
    expected = isinstance(exc, (SourceError, RetryableError))
    logger.error("etl failed %s: %s", what, exc, exc_info=not expected)
    return "failed"


@shared_task(bind=True, max_retries=None, name="etl.fetch_day")
def fetch_day(self: Task[Any, Any], source: str, dataset: str, day: str) -> str:
    """One dataset for one session; its own task, so a failure blocks nothing else."""
    what = f"{source}/{dataset} {day}"
    session = date.fromisoformat(day)
    try:
        run = build_adapter(source).run(dataset, session, session)
    except SourceNotReadyError as exc:
        return _not_ready(self, what, exc)
    except Exception as exc:
        return _failed(what, exc)
    _recheck_if_blocked(session)
    return f"success: {run.rows} rows"


@shared_task(name="etl.run_batch")
def run_batch(batch: str) -> list[str]:
    """Queue a batch for today's session and the sessions before it that some datasets redo."""
    session = _session_today(BATCH_SLOTS[batch])
    if session is None:
        return []
    calendar, today = session
    queued = []
    for job in BATCHES[batch]:
        for back in job.sessions_back:
            day = calendar.shift(today, -back).isoformat()
            fetch_day.delay(job.source, job.dataset, day)
            queued.append(f"{job.source}/{job.dataset} {day}")
    return queued


@shared_task(name="etl.prepare_day")
def prepare_day() -> list[str]:
    """Correct the calendar and fetch what earlier sessions missed, before today's run."""
    session = _session_today()
    if session is None:
        return []
    today = session[1]
    try:
        call_command("refresh_calendar")
    except Exception as exc:
        _failed("trading_calendar refresh", exc)
    calendar = TradingDays.from_db()
    since = today - timedelta(days=settings.ETL_CATCHUP_LOOKBACK_DAYS)
    until = calendar.latest_on_or_before(today - timedelta(days=1))
    report = catch_up(max(since, calendar.start), until, calendar)
    for line in report.lines():
        logger.info("catchup %s", line)
    if not DataQualityLog.objects.filter(date=until, check_name__in=CHECKS).exists():
        _check(until, calendar)
    return report.lines()


@shared_task(bind=True, max_retries=None, name="etl.refresh_master")
def refresh_master(self: Task[Any, Any]) -> str:
    """Update stock_master once today's quotes of both exchanges are stored."""
    session = _session_today("REFRESH_MASTER")
    if session is None:
        return "closed"
    calendar, today = session
    try:
        refresh_stock_master(today, calendar)
    except SourceNotReadyError as exc:
        return _not_ready(self, f"stock_master {today}", exc)
    except Exception as exc:
        return _failed(f"stock_master {today}", exc)
    return "success"


@shared_task(name="etl.check_quality")
def check_quality() -> int:
    """The five daily checks after the evening ETL, plus which daily datasets did not land."""
    session = _session_today("CHECK_QUALITY")
    if session is None:
        return 0
    calendar, today = session
    results = run_checks(today, calendar)
    missing = sorted(
        f"{source}/{dataset}"
        for source, dataset in _daily_jobs()
        if today not in covered_days(source, dataset, [today])
    )
    severity = DataQualityLog.Severity.WARNING if missing else DataQualityLog.Severity.INFO
    landed = _replace_log(today, ETL_RUNS_CHECK, severity, {"missing": missing})
    dispatch([*results, landed])
    return len(results)


@shared_task(name="etl.us_close")
def us_close() -> str:
    """US closes for the last session, once; a series without them keeps its last value, logged."""
    calendar = TradingDays.from_db()
    day = calendar.latest_on_or_before(timezone.localtime().date() - timedelta(days=1))
    if DataQualityLog.objects.filter(date=day, check_name=US_CLOSE_CHECK).exists():
        return "done"
    adapter = build_adapter("yahoo", calendar)
    for dataset in US_SERIES:
        try:
            adapter.run(dataset, day, day)
        except Exception as exc:
            _failed(f"yahoo/{dataset} {day}", exc)
    stale = {}
    for index_id in US_SERIES.values():
        expected = calendar.shift(day, -US_LAG.get(index_id, 0))
        seen = _latest(index_id)
        if seen is None or seen < expected:
            stale[index_id] = seen.isoformat() if seen is not None else ""
    severity = DataQualityLog.Severity.WARNING if stale else DataQualityLog.Severity.INFO
    dispatch([_replace_log(day, US_CLOSE_CHECK, severity, {"stale": stale})])
    return severity


def _replace_log(day: date, check: str, severity: str, detail: dict[str, Any]) -> DataQualityLog:
    with transaction.atomic():
        DataQualityLog.objects.filter(date=day, check_name=check).delete()
        return DataQualityLog.objects.create(
            date=day, check_name=check, severity=severity, detail=detail
        )


def _latest(index_id: str) -> date | None:
    rows = IndexDaily.objects.filter(index_id=index_id).order_by("-date")
    return rows.values_list("date", flat=True).first()


@shared_task(bind=True, max_retries=None, name="etl.weekly_holding")
def weekly_holding(self: Task[Any, Any]) -> str:
    """TDCC's file of the last ended week, asked for with its last session, even when run late."""
    calendar = TradingDays.from_db()
    today = timezone.localtime().date()
    saturday = today - timedelta(days=(today.weekday() - SATURDAY) % 7)
    sessions = calendar.between(saturday - timedelta(days=SATURDAY), saturday)
    if not sessions:
        return "no session this week"
    return _run_once(self, "tdcc", "weekly_holding", sessions[-1], calendar)


@shared_task(bind=True, max_retries=None, name="etl.business_signal")
def business_signal(self: Task[Any, Any]) -> str:
    """data.gov.tw's monthly file on its due day, which need not be a session."""
    today = timezone.localtime().date()
    return _run_once(self, "datagov", "business_signal", today, TradingDays.from_db())


def _run_once(
    task: Task[Any, Any], source: str, dataset: str, day: date, calendar: TradingDays
) -> str:
    what = f"{source}/{dataset} {day}"
    try:
        run = build_adapter(source, calendar).run(dataset, day, day)
    except SourceNotReadyError as exc:
        return _not_ready(task, what, exc)
    except Exception as exc:
        return _failed(what, exc)
    return f"success: {run.rows} rows"


def _daily_jobs() -> set[tuple[str, str]]:
    return {(job.source, job.dataset) for jobs in BATCHES.values() for job in jobs}


def etl_complete(day: date) -> bool:
    """Every dataset of the daily batches has a run covering the session."""
    return all(day in covered_days(source, dataset, [day]) for source, dataset in _daily_jobs())


def ready_for_research(day: date) -> bool:
    """The research jobs of a session run only after its ETL and quality gate both pass."""
    return etl_complete(day) and should_run_prediction(day)
