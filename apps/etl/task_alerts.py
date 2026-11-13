"""Alerts for Celery tasks that fail, or swallow an error they did not expect (SPEC §5)."""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import redis
import requests
from celery import current_task
from celery.signals import task_failure
from django.conf import settings
from django.db import DatabaseError, InterfaceError, OperationalError, transaction
from django.utils import timezone

from apps.etl.models import DataQualityLog
from apps.market.calendar import CalendarError

logger = logging.getLogger(__name__)
CHECK = "task_failure"
SIGNAL_UID = "apps.etl.task_alerts"
DATABASE = "database"
UNREACHABLE = (OperationalError, InterfaceError)
KEY_TTL = 36 * 3600
STORE_TIMEOUT = 2.0
TEXT_LIMIT = 200


def connect() -> None:
    """Register the handler once; nothing is contacted until a task fails."""
    task_failure.connect(on_task_failure, dispatch_uid=SIGNAL_UID)


def on_task_failure(
    sender: Any = None,
    exception: BaseException | None = None,
    args: Any = None,
    kwargs: Any = None,
    **extra: Any,
) -> None:
    """Celery's task_failure: an error no task handled once retries ran out, or a lost worker."""
    with _contained():
        name = str(getattr(sender, "name", "") or "unknown task")
        context = f"args={repr(args or ())[:TEXT_LIMIT]} kwargs={repr(kwargs or {})[:TEXT_LIMIT]}"
        _report(name, exception or RuntimeError("no exception given"), context)


def report_failure(name: str, exc: BaseException, context: str = "") -> None:
    """Log, post and count one failure; never raises, whatever breaks on the way."""
    with _contained():
        _report(name, exc, context)


def report_swallowed(exc: BaseException, context: str) -> None:
    """For an error a task caught and outlived: Celery saw a success, so no signal follows."""
    report_failure(str(current_task.name) if current_task else "etl", exc, context)


@contextmanager
def _contained() -> Iterator[None]:
    """Nothing that breaks while alerting may reach the task."""
    try:
        yield
    except Exception as error:
        logger.error("task failure alert broke: %s", type(error).__name__)


def _report(name: str, exc: BaseException, context: str) -> None:
    """The first failure of a task, day and error class is posted; a dead database is one task."""
    now = timezone.localtime()
    kind = type(exc).__name__
    unreachable = isinstance(exc, UNREACHABLE)
    subject = DATABASE if unreachable else name
    scope = DATABASE if unreachable else f"{name}:{kind}"
    key = f"alert:{CHECK}:{now.date().isoformat()}:{scope}"
    first = _claim(key)
    logger.error("task failed %s (%s) %s%s", name, kind, context, "" if first else "; repeated")
    if first and not _post(name, exc, context, now):
        _release(key)
    if not unreachable:
        _record(now, subject, kind)


def _claim(key: str) -> bool:
    """True for the first claim of a key; a store that cannot answer never silences an alert."""
    return _ask(lambda store: bool(store.set(key, "1", nx=True, ex=KEY_TTL)), True)


def _release(key: str) -> None:
    """An alert that did not arrive gives its key back, so the next failure posts again."""
    _ask(lambda store: bool(store.delete(key)), False)


def _ask(question: Callable[[Any], bool], otherwise: bool) -> bool:
    """One question to the store, or the fallback when there is none or it cannot answer."""
    url = str(settings.TASK_ALERT_REDIS_URL)
    if not url:
        return otherwise
    try:
        store = redis.Redis.from_url(
            url, socket_connect_timeout=STORE_TIMEOUT, socket_timeout=STORE_TIMEOUT
        )
        try:
            return question(store)
        finally:
            store.close()
    except Exception as error:
        logger.error("task failure alert could not reach its store: %s", type(error).__name__)
        return otherwise


def _post(name: str, exc: BaseException, context: str, now: datetime) -> bool:
    """False when the webhook did not take it; only calendar errors carry their message."""
    url = str(settings.ALERT_WEBHOOK_URL)
    if not url:
        return True
    kind = type(exc).__name__
    message = str(exc)[:TEXT_LIMIT] if isinstance(exc, CalendarError) else None
    detail = {
        "task": name,
        "error": kind,
        "message": message,
        "context": context,
        "first_at": now.isoformat(timespec="seconds"),
    }
    alert = {
        "date": now.date().isoformat(),
        "check": CHECK,
        "severity": DataQualityLog.Severity.WARNING.value,
        "detail": detail,
    }
    what = f"{kind}: {message}" if message else kind
    body = {"text": _headline(name, what, context, isinstance(exc, UNREACHABLE)), "alerts": [alert]}
    try:
        response = requests.post(url, json=body, timeout=float(settings.ALERT_TIMEOUT))
        response.raise_for_status()
    except requests.RequestException as error:
        logger.error("task failure alert webhook failed: %s", type(error).__name__)
        return False
    return True


def _headline(name: str, what: str, context: str, unreachable: bool) -> str:
    """All that a channel showing only the text gets to see."""
    if unreachable:
        seen = f"database error ({what}), first seen in {name} {context}".rstrip()
        return f"task failed: {seen}; one alert a day covers every task"
    return f"task failed: {name} ({what}) {context}".rstrip()


def _record(now: datetime, subject: str, kind: str) -> None:
    """Best effort: the alert has already gone out, so a failing write is only logged."""
    try:
        _count(now, subject, kind)
    except DatabaseError as error:
        logger.error("task failure alert could not record: %s", type(error).__name__)


def _count(now: datetime, subject: str, kind: str) -> None:
    """The day's one task_failure row, a warning so that it never closes the research gate."""
    with transaction.atomic():
        rows = DataQualityLog.objects.select_for_update().filter(date=now.date(), check_name=CHECK)
        earlier = rows.first()
        counted = dict(earlier.detail["tasks"]) if earlier is not None else {}
        seen = counted.get(subject, {"count": 0, "first_at": now.isoformat(timespec="seconds")})
        counted[subject] = {
            "count": int(seen["count"]) + 1,
            "first_at": seen["first_at"],
            "errors": sorted({*seen.get("errors", []), kind}),
        }
        rows.delete()
        DataQualityLog.objects.create(
            date=now.date(),
            check_name=CHECK,
            severity=DataQualityLog.Severity.WARNING,
            detail={"tasks": counted},
        )
