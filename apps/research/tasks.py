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
from django.db.models import Max
from django.utils import timezone

from apps.etl.alerts import dispatch
from apps.etl.models import DataQualityLog
from apps.etl.tasks import US_CLOSE_CHECK, ready_for_research
from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.research.lock import ResearchBusyError, research_lock
from apps.research.models import HoldoutScope, LabelDaily
from apps.research.monitor import run_monitor
from apps.research.predict import predict
from apps.research.production import DESIGNS, build_run
from apps.research.promotion import PROMOTION_CHECK, first_day, promote

logger = logging.getLogger(__name__)
EVENING_CHECK = "research_evening"
MORNING_CHECK = "research_morning"
RETRAIN_CHECK = "research_retrain"
NOT_FROZEN = "the design is not frozen: DESIGNS names no research_run_key"
MISSED = "missed"
RETRAIN_WINDOW = 5
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


def retrain_key(scope: str, day: date) -> str:
    return f"{scope}-{day:%Y-%m}"


def _month(check: str, day: date) -> Any:
    return DataQualityLog.objects.filter(
        check_name=check, date__year=day.year, date__month=day.month
    )


def _pending(day: date) -> list[str]:
    """The run keys this month's retrain built that no promotion has tried yet."""
    row = _month(RETRAIN_CHECK, day).filter(date__lte=day).order_by("-date", "-pk").first()
    built = {} if row is None else row.detail.get("built", {})
    keys = [built[scope] for scope in HoldoutScope.values if scope in built]
    tried = DataQualityLog.objects.filter(check_name=PROMOTION_CHECK, detail__run_key__in=keys)
    seen = {row.detail["run_key"] for row in tried}
    return [key for key in keys if key not in seen]


def _promote(day: date) -> str:
    """Promote each untried build from its first day; a refusal fails the step after the rest."""
    done, failed = [], []
    for run_key in _pending(day):
        try:
            result = promote(run_key, first_day(run_key), settings.FEATURE_PARQUET_DIR)
        except Exception as error:
            failed.append(f"{run_key}: {type(error).__name__}: {error}")
            continue
        done.append(f"{run_key}: {'promoted' if result.promoted else 'kept the serving runs'}")
    if failed:
        raise RuntimeError("; ".join([*failed, *done]))
    return "; ".join(done)


STEPS: dict[str, Callable[[date], str]] = {
    "labels": _labels,
    "features": _features,
    "predict": _predict,
    "monitor": _monitor,
    "promote": _promote,
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
    session = date.fromisoformat(day)
    severity = Severity.WARNING if extra else Severity.INFO
    _record(session, check, severity, {"steps": done, **extra})
    if check == MORNING_CHECK and _retrain_due(session):
        retrain.delay(day)
    return str(severity)


def _retrain_due(day: date) -> bool:
    """A month retrains once, from a morning among its first sessions."""
    calendar = TradingDays.from_db()
    opening = calendar.between(day.replace(day=1), day)
    return len(opening) <= RETRAIN_WINDOW and not _month(RETRAIN_CHECK, day).exists()


@shared_task(name="research.retrain")
def retrain(day: str) -> str:
    """This month's N and N′ of every frozen scope as of day, once the labels reach day."""
    session = date.fromisoformat(day)
    last = LabelDaily.objects.aggregate(last=Max("date"))["last"]
    built: dict[str, str] = {}
    refused: dict[str, str] = {}
    failed: dict[str, str] = {}
    for scope in HoldoutScope.values:
        research = DESIGNS[scope].research_run_key
        if last is None or last < session:
            refused[scope] = f"label_daily ends on {last}, before {session}"
        elif research is None:
            refused[scope] = NOT_FROZEN
        else:
            key = retrain_key(scope, session)
            try:
                build_run(scope, key, research, session, settings.FEATURE_PARQUET_DIR)
            except Exception as error:
                logger.error("research retrain %s %s failed: %s", scope, day, error)
                failed[scope] = f"{type(error).__name__}: {error}"
            else:
                built[scope] = key
    severity = Severity.WARNING if refused or failed else Severity.INFO
    _record(
        session, RETRAIN_CHECK, severity, {"built": built, "refused": refused, "failed": failed}
    )
    return str(severity)


def _flag_missed(calendar: TradingDays, day: date) -> None:
    """Warn when the session before day never had a morning run, once the schedule has run."""
    try:
        before = calendar.shift(day, -1)
    except OutsideCalendarError:
        return
    seen = DataQualityLog.objects.filter(check_name=MORNING_CHECK)
    if seen.filter(date=before).exists() or not seen.filter(date__lt=before).exists():
        return
    _record(before, MORNING_CHECK, Severity.WARNING, {"reason": MISSED})


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
        calendar = TradingDays.from_db()
        session = calendar.latest_on_or_before(timezone.localtime().date() - timedelta(days=1))
        _flag_missed(calendar, session)
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
    names = (*names, "promote") if _pending(session) else names
    _start(MORNING_CHECK, session, names, {} if landed else {"us_close": US_CLOSE_MISSING})
    return "started"
