"""Celery tasks behind the 23:00 and 06:00 research schedule (SPEC §5 每日增量)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, time, timedelta
from io import StringIO
from typing import Any

from celery import Task, chain, shared_task
from django.conf import settings
from django.core.management import call_command
from django.db import transaction
from django.utils import timezone

from apps.etl.alerts import dispatch
from apps.etl.models import DataQualityLog
from apps.etl.tasks import US_CLOSE_CHECK, ready_for_research
from apps.market.calendar import TradingDays
from apps.research.lock import ResearchBusyError, research_lock
from apps.research.monitor import run_monitor
from apps.research.predict import predict

logger = logging.getLogger(__name__)
EVENING_CHECK = "research_evening"
MORNING_CHECK = "research_morning"
GATE_SHUT = "research gate shut"
US_CLOSE_MISSING = "missing"
EVENING_STEPS = ("features", "labels")
MORNING_STEPS = ("features", "predict", "monitor")
Severity = DataQualityLog.Severity
Done = dict[str, str]


def _command(name: str, **options: Any) -> str:
    out = StringIO()
    call_command(name, stdout=out, **options)
    return out.getvalue().strip()


def _features(day: date) -> str:
    return _command("build_features", date=day)


def _labels(day: date) -> str:
    return _command("build_labels", end=day)


def _predict(day: date) -> str:
    result = predict(day, settings.FEATURE_PARQUET_DIR)
    written = [f"{run_id}: {count} rows" for run_id, count in result.written.items()]
    return "; ".join([*written, *(f"no serving run for {slot}" for slot in result.missing)])


def _monitor(day: date) -> str:
    result = run_monitor(day, TradingDays.from_db())
    degraded = ", ".join(result.degraded) or "none"
    return f"{result.scored} model_monitor rows, degraded {degraded}"


STEPS: dict[str, Callable[[date], str]] = {
    "labels": _labels,
    "features": _features,
    "predict": _predict,
    "monitor": _monitor,
}


def _record(day: date, check: str, severity: str, detail: dict[str, Any]) -> None:
    """One row per day and check, replaced on rerun; warnings are sent as alerts."""
    with transaction.atomic():
        DataQualityLog.objects.filter(date=day, check_name=check).delete()
        row = DataQualityLog.objects.create(
            date=day, check_name=check, severity=severity, detail=detail
        )
    if severity != Severity.INFO:
        dispatch([row])


def _finished(day: date, check: str) -> bool:
    return DataQualityLog.objects.filter(
        date=day, check_name=check, severity=Severity.INFO
    ).exists()


def _us_close_landed(day: date) -> bool:
    return DataQualityLog.objects.filter(date=day, check_name=US_CLOSE_CHECK).exists()


def _early(slot: str) -> bool:
    """A run fired before its slot, as beat does on waking past midnight, is skipped and logged."""
    now = timezone.localtime()
    if now.time() >= time.fromisoformat(settings.RESEARCH_AT[slot]):
        return False
    logger.warning("research skipped %s: fired at %s, before its slot", slot, now.isoformat())
    return True


def _failed(check: str, day: date, name: str, error: Exception, done: list[Done]) -> None:
    logger.error("research %s %s failed: %s", name, day, error)
    detail = {"failed": name, "error": f"{type(error).__name__}: {error}", "steps": done}
    _record(day, check, Severity.WARNING, detail)


def _start(check: str, day: date, names: tuple[str, ...], extra: dict[str, str]) -> None:
    """Queue the steps as one chain; each starts only once the one before it succeeded."""
    iso = day.isoformat()
    first, *rest = names
    chain(
        step.s([], first, check, iso),
        *(step.s(name, check, iso) for name in rest),
        finish.s(check, iso, extra),
    ).delay()


@shared_task(bind=True, max_retries=None, name="research.step")
def step(self: Task[Any, Any], done: list[Done], name: str, check: str, day: str) -> list[Done]:
    """One chain step under the research lock; a busy lock retries, a failure stops the chain."""
    session = date.fromisoformat(day)
    try:
        with research_lock(f"{name} {day}"):
            summary = STEPS[name](session)
    except ResearchBusyError as error:
        if self.request.retries < settings.RESEARCH_LOCK_RETRIES:
            raise self.retry(exc=error, countdown=settings.RESEARCH_LOCK_COUNTDOWN) from error
        _failed(check, session, name, error, done)
        raise
    except Exception as error:
        _failed(check, session, name, error, done)
        raise
    return [*done, {"step": name, "result": summary}]


@shared_task(name="research.finish")
def finish(done: list[Done], check: str, day: str, extra: dict[str, str]) -> str:
    """Record the chain as done, as a warning when it ran without the US close."""
    severity = Severity.WARNING if extra else Severity.INFO
    _record(date.fromisoformat(day), check, severity, {"steps": done, **extra})
    return str(severity)


@shared_task(bind=True, max_retries=None, name="research.evening")
def evening(self: Task[Any, Any], day: str | None = None) -> str:
    """Today's features and the recent labels, once the session's research gate passes."""
    if not settings.RESEARCH_SCHEDULE_ENABLED:
        return "disabled"
    if day is None and _early("EVENING"):
        return "early"
    session = timezone.localtime().date() if day is None else date.fromisoformat(day)
    if not TradingDays.from_db().is_trading_day(session):
        return "closed"
    if _finished(session, EVENING_CHECK):
        return "done"
    if not ready_for_research(session):
        if self.request.retries < settings.ETL_NOT_READY_RETRIES:
            raise self.retry(
                args=(session.isoformat(),), countdown=settings.ETL_NOT_READY_COUNTDOWN
            )
        _record(session, EVENING_CHECK, Severity.WARNING, {"reason": GATE_SHUT})
        return "not ready"
    _start(EVENING_CHECK, session, EVENING_STEPS, {})
    return "started"


@shared_task(bind=True, max_retries=None, name="research.morning")
def morning(self: Task[Any, Any], day: str | None = None) -> str:
    """Before the open: the last session's features with the US close, predictions, monitor."""
    if not settings.RESEARCH_SCHEDULE_ENABLED:
        return "disabled"
    if day is None and _early("MORNING"):
        return "early"
    if day is None:
        yesterday = timezone.localtime().date() - timedelta(days=1)
        session = TradingDays.from_db().latest_on_or_before(yesterday)
    else:
        session = date.fromisoformat(day)
    if _finished(session, MORNING_CHECK):
        return "done"
    if not ready_for_research(session):
        _record(session, MORNING_CHECK, Severity.WARNING, {"reason": GATE_SHUT})
        return "gate shut"
    landed = _us_close_landed(session)
    if not landed and self.request.retries < settings.RESEARCH_US_CLOSE_RETRIES:
        raise self.retry(
            args=(session.isoformat(),), countdown=settings.RESEARCH_US_CLOSE_COUNTDOWN
        )
    names = MORNING_STEPS if _finished(session, EVENING_CHECK) else ("labels", *MORNING_STEPS)
    _start(MORNING_CHECK, session, names, {} if landed else {"us_close": US_CLOSE_MISSING})
    return "started"
