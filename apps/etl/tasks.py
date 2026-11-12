"""Celery tasks behind the daily ETL schedule (SPEC §5 每日增量, 重試, 失敗處理)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, time, timedelta
from typing import Any, NamedTuple

from celery import Task, shared_task
from celery.schedules import crontab
from django.conf import settings
from django.core.management import call_command
from django.db import transaction
from django.db.models import Min
from django.utils import timezone

from apps.etl.adapters import build_adapter
from apps.etl.adapters.corporate_actions import DIVIDEND_EVENT
from apps.etl.adapters.exchange_notices import DIVIDEND_NOTICE
from apps.etl.adapters.mops import REVENUE_SCAN, STATEMENT_SCAN
from apps.etl.alerts import dispatch
from apps.etl.calendar_upkeep import keep_up
from apps.etl.catchup import REDONE, catch_up
from apps.etl.errors import SourceError, SourceNotReadyError
from apps.etl.master_refresh import DAILY_DATASET, refresh_stock_master
from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.quality import (
    CHECKS,
    UNANNOUNCED,
    alerting,
    run_checks,
    should_run_prediction,
    unannounced_result,
)
from apps.etl.ratelimit import RetryableError, ThrottledError
from apps.etl.revenue_scan import open_months
from apps.etl.runs import covered_days
from apps.etl.statement_summary import open_periods
from apps.etl.trading_calendar import REFRESH_DATASET, REFRESH_SOURCE
from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.market.models import IndexDaily, IndexId

logger = logging.getLogger(__name__)
US_SERIES = {
    "sox": IndexId.SOX.value,
    "tsm_adr": IndexId.TSM_ADR.value,
    "usdtwd": IndexId.USDTWD.value,
}
US_CLOSE_CHECK = "us_close"
ETL_RUNS_CHECK = "etl_runs"
AUDITED = (*CHECKS, ETL_RUNS_CHECK, UNANNOUNCED)
GATE_DATASETS = frozenset(
    {
        "daily_price",
        "index_daily",
        "institutional",
        "margin",
        "sbl",
        "foreign_holding",
        "daily_valuation",
        DIVIDEND_EVENT,
    }
)
Severity = DataQualityLog.Severity
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
CARRIED_FORWARD = frozenset(
    {
        ("stock_master", DAILY_DATASET),
        ("datagov", "business_signal"),
        *((source, DIVIDEND_NOTICE) for source in EXCHANGES),
    }
)


class Due(NamedTuple):
    """A run the evening check expects, of one source and dataset, covering one day."""

    source: str
    dataset: str
    day: date


class Fetch(NamedTuple):
    """One dataset of one source, run for today and the given sessions before it."""

    source: str
    dataset: str
    sessions_back: tuple[int, ...] = (0,)


def _both(*datasets: str, sessions_back: tuple[int, ...] = (0,)) -> list[Fetch]:
    return [Fetch(source, name, sessions_back) for name in datasets for source in EXCHANGES]


BATCHES: dict[str, list[Fetch]] = {
    "prices": [*_both("daily_price", "index_daily", DIVIDEND_EVENT), Fetch("twse", "return_index")],
    "afternoon": _both(
        "institutional", "daily_valuation", "attention", "disposition", "full_delivery"
    ),
    "qfii": [Fetch("tpex", "foreign_holding"), Fetch("tpex", "shares_outstanding")],
    "night": [
        *_both("margin", "sbl"),
        *_both("day_trade", sessions_back=tuple(range(REDONE["day_trade"] + 1))),
        Fetch("twse", "foreign_holding"),
        Fetch("twse", "shares_outstanding"),
        *_both(DIVIDEND_NOTICE),
    ],
    "notices": _both("attention", "disposition"),
}


def _before_slot(slot: str) -> bool:
    """A run fired before its slot, as beat does on waking past midnight, is skipped and logged."""
    now = timezone.localtime()
    if now.time() >= time.fromisoformat(settings.ETL_AT[slot]):
        return False
    logger.warning("etl skipped %s: fired at %s, before its slot", slot, now.isoformat())
    return True


def _session_today(slot: str | None = None) -> tuple[TradingDays, date] | None:
    """Today when it is a session and its slot has come."""
    if slot is not None and _before_slot(slot):
        return None
    calendar = TradingDays.from_db()
    today = timezone.localtime().date()
    return (calendar, today) if calendar.is_trading_day(today) else None


def _landed(day: date, owned: frozenset[Due] = frozenset()) -> DataQualityLog:
    """The daily datasets and owned runs without a run covering them; a gate one is an error."""
    wanted = {Due(source, dataset, day) for source, dataset in _daily_jobs()} | owned
    missing = sorted({f"{due.source}/{due.dataset}" for due in wanted if not _covered(due)})
    blocking = [key for key in missing if key.split("/")[1] in GATE_DATASETS]
    severity = Severity.ERROR if blocking else Severity.WARNING if missing else Severity.INFO
    return _replace_log(day, ETL_RUNS_CHECK, severity, {"missing": missing, "blocking": blocking})


def _covered(due: Due) -> bool:
    """A run covers the day; for snapshots without history, any later success carries it too."""
    if due.day in covered_days(due.source, due.dataset, [due.day]):
        return True
    return (due.source, due.dataset) in CARRIED_FORWARD and EtlRun.objects.filter(
        source=due.source,
        dataset=due.dataset,
        status=EtlRun.Status.SUCCESS,
        start_date__gt=due.day,
    ).exists()


def _audit(
    day: date, calendar: TradingDays, owned: frozenset[Due] = frozenset()
) -> list[DataQualityLog]:
    results = [*run_checks(day, calendar), _landed(day, owned), _unannounced(day)]
    dispatch(results)
    return results


def _unannounced(day: date) -> DataQualityLog:
    """Kept apart from CHECKS: it only warns, so the research gate never waits on it."""
    found = unannounced_result(day)
    return _replace_log(day, UNANNOUNCED, found.severity, found.detail)


def _alerts(day: date) -> set[tuple[str, str]]:
    rows = DataQualityLog.objects.filter(date=day, check_name__in=AUDITED)
    return set(rows.exclude(severity=Severity.INFO).values_list("check_name", "severity"))


def _reaudit(day: date, calendar: TradingDays | None = None) -> None:
    """Data landing after a session's audit redoes what still alerts; only new alerts go out."""
    before = _alerts(day)
    results: list[DataQualityLog] = []
    if any(check in CHECKS and level == Severity.ERROR for check, level in before):
        results.extend(run_checks(day, calendar or TradingDays.from_db()))
    if any(check == ETL_RUNS_CHECK for check, _ in before):
        results.append(_landed(day, _still_listed(day, calendar or TradingDays.from_db())))
    if any(check == UNANNOUNCED for check, _ in before):
        results.append(_unannounced(day))
    fresh = [row for row in alerting(results) if (row.check_name, row.severity) not in before]
    if fresh:
        dispatch(fresh)


def _retry_later(task: Task[Any, Any], what: str, exc: Exception) -> str:
    """Unpublished days and network failures retry while retries last; throttling never does."""
    if isinstance(exc, ThrottledError):
        return _failed(what, exc)
    if task.request.retries >= settings.ETL_NOT_READY_RETRIES:
        logger.error("etl gave up %s after %d retries: %s", what, task.request.retries, exc)
        return "not ready" if isinstance(exc, SourceNotReadyError) else "unreachable"
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
    except (SourceNotReadyError, RetryableError) as exc:
        return _retry_later(self, what, exc)
    except Exception as exc:
        return _failed(what, exc)
    _reaudit(session)
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
    """Keep the calendar ahead and correct it, fetch what sessions missed, then audit them."""
    try:
        keep_up(timezone.localtime().date())
    except Exception as exc:
        _failed("trading_calendar upkeep", exc)
    session = _session_today()
    if session is None:
        return []
    today = session[1]
    try:
        call_command("refresh_calendar")
    except Exception as exc:
        _failed("trading_calendar refresh", exc)
    calendar = TradingDays.from_db()
    since = max(today - timedelta(days=settings.ETL_CATCHUP_LOOKBACK_DAYS), calendar.start)
    until = calendar.latest_on_or_before(today - timedelta(days=1))
    report = catch_up(since, until, calendar)
    for line in report.lines():
        logger.info("catchup %s", line)
    _audit_missed(calendar.between(since, until), calendar)
    return report.lines()


def _audit_missed(sessions: Sequence[date], calendar: TradingDays) -> None:
    """Audit sessions the evening audit missed since the schedule began; redo the rest."""
    first = DataQualityLog.objects.filter(check_name=ETL_RUNS_CHECK).aggregate(first=Min("date"))
    checked = set(
        DataQualityLog.objects.filter(date__in=sessions, check_name__in=CHECKS).values_list(
            "date", flat=True
        )
    )
    for day in sessions:
        if day in checked:
            _reaudit(day, calendar)
        elif day == sessions[-1] or (first["first"] is not None and day >= first["first"]):
            _audit(day, calendar)


@shared_task(bind=True, max_retries=None, name="etl.refresh_master")
def refresh_master(self: Task[Any, Any]) -> str:
    """Update stock_master once today's quotes of both exchanges are stored."""
    session = _session_today("REFRESH_MASTER")
    if session is None:
        return "closed"
    calendar, today = session
    try:
        refresh_stock_master(today, calendar)
    except (SourceNotReadyError, RetryableError) as exc:
        return _retry_later(self, f"stock_master {today}", exc)
    except Exception as exc:
        return _failed(f"stock_master {today}", exc)
    return "success"


@shared_task(bind=True, max_retries=None, name="etl.statement_scan")
def statement_scan(self: Task[Any, Any], slot: str) -> str:
    """MOPS summaries on sessions while a quarter is being filed; outside the research gate."""
    session = _session_today(slot)
    if session is None:
        return "closed"
    calendar, today = session
    if not open_periods(today, settings.STATEMENT_SCAN_DAYS_AFTER):
        return "no period open"
    return _run_once(self, "mops", STATEMENT_SCAN, today, calendar)


@shared_task(bind=True, max_retries=None, name="etl.revenue_scan")
def revenue_scan(self: Task[Any, Any], slot: str) -> str:
    """MOPS monthly revenue pages on sessions while a month is being filed; outside the gate."""
    session = _session_today(slot)
    if session is None:
        return "closed"
    calendar, today = session
    if not open_months(today, settings.REVENUE_SCAN_DAYS_AFTER):
        return "no month open"
    return _run_once(self, "mops", REVENUE_SCAN, today, calendar)


@shared_task(name="etl.check_quality")
def check_quality() -> int:
    """The five daily checks after the evening ETL, plus which expected runs did not land."""
    session = _session_today("CHECK_QUALITY")
    if session is None:
        return 0
    calendar, today = session
    return len(_audit(today, calendar, _scans_due(today) | _periodic_due(today, calendar)))


@shared_task(name="etl.us_close")
def us_close() -> str:
    """US closes for the last session, once; a series without them keeps its last value, logged."""
    if _before_slot("US_CLOSE"):
        return "early"
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
    severity = Severity.WARNING if stale else Severity.INFO
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
    """data.gov.tw's monthly file, recorded on its due day even when beat fires it late."""
    today = timezone.localtime().date()
    due = [day for day in signal_days() if day <= today.day]
    on = today.replace(day=max(due)) if due else today
    return _run_once(self, "datagov", "business_signal", on, TradingDays.from_db())


def _run_once(
    task: Task[Any, Any], source: str, dataset: str, day: date, calendar: TradingDays
) -> str:
    what = f"{source}/{dataset} {day}"
    try:
        run = build_adapter(source, calendar).run(dataset, day, day)
    except (SourceNotReadyError, RetryableError) as exc:
        return _retry_later(task, what, exc)
    except Exception as exc:
        return _failed(what, exc)
    return f"success: {run.rows} rows"


def _scans_due(day: date) -> frozenset[Due]:
    """MOPS scans the session should have run; only its own evening check can still act on them."""
    due = {
        STATEMENT_SCAN: bool(open_periods(day, settings.STATEMENT_SCAN_DAYS_AFTER)),
        REVENUE_SCAN: bool(open_months(day, settings.REVENUE_SCAN_DAYS_AFTER)),
    }
    return frozenset(Due("mops", dataset, day) for dataset, open_now in due.items() if open_now)


def signal_days() -> set[int]:
    """Days of the month the beat schedule fetches the business signal on."""
    schedule = settings.CELERY_BEAT_SCHEDULE["etl-business-signal"]["schedule"]
    return set(getattr(schedule, "day_of_month", ())) if isinstance(schedule, crontab) else set()


def _periodic_due(day: date, calendar: TradingDays) -> frozenset[Due]:
    """Master and calendar refreshes, and weekly and monthly files due since the last session."""
    due = {
        Due("stock_master", DAILY_DATASET, day),
        Due(REFRESH_SOURCE, REFRESH_DATASET, day - timedelta(days=1)),
    }
    try:
        previous = calendar.shift(day, -1)
    except OutsideCalendarError:
        return frozenset(due)
    for n in range(1, (day - previous).days + 1):
        passed = previous + timedelta(days=n)
        if passed.weekday() == SATURDAY:
            week = calendar.between(max(passed - timedelta(days=SATURDAY), calendar.start), passed)
            if week:
                due.add(Due("tdcc", "weekly_holding", week[-1]))
        if passed.day in signal_days():
            due.add(Due("datagov", "business_signal", passed))
    return frozenset(due)


def _still_listed(day: date, calendar: TradingDays) -> frozenset[Due]:
    """Periodic runs the session's own check listed; a later audit keeps them until they land."""
    row = DataQualityLog.objects.filter(date=day, check_name=ETL_RUNS_CHECK).first()
    listed = set(row.detail.get("missing", [])) if row is not None else set()
    return frozenset(
        due for due in _periodic_due(day, calendar) if f"{due.source}/{due.dataset}" in listed
    )


def _daily_jobs() -> set[tuple[str, str]]:
    return {(job.source, job.dataset) for jobs in BATCHES.values() for job in jobs}


def etl_complete(day: date) -> bool:
    """Every daily snapshot the session's features read has a run covering it."""
    return all(
        day in covered_days(source, dataset, [day])
        for source, dataset in _daily_jobs()
        if dataset in GATE_DATASETS
    )


def ready_for_research(day: date) -> bool:
    """The research jobs of a session run only after its ETL and quality gate both pass."""
    return etl_complete(day) and should_run_prediction(day)
