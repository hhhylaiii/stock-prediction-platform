"""Alerts for Celery tasks that fail, or swallow an error they did not expect (SPEC §5)."""

import logging
from datetime import datetime
from typing import Any

import redis
import requests
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
    name = str(getattr(sender, "name", "") or "unknown task")
    context = f"args={repr(args or ())[:TEXT_LIMIT]} kwargs={repr(kwargs or {})[:TEXT_LIMIT]}"
    report_failure(name, exception or RuntimeError("no exception given"), context)


def report_failure(name: str, exc: BaseException, context: str = "") -> None:
    """Log, post and count one failure; never raises, whatever breaks on the way."""
    try:
        _report(name, exc, context)
    except Exception as error:
        logger.error("task failure alert broke: %s", type(error).__name__)


def _report(name: str, exc: BaseException, context: str) -> None:
    """The first failure of a task, day and error class is posted; a dead database is one task."""
    now = timezone.localtime()
    kind = type(exc).__name__
    unreachable = isinstance(exc, UNREACHABLE)
    subject = DATABASE if unreachable else name
    first = _claim(f"alert:{CHECK}:{now.date().isoformat()}:{subject}:{kind}")
    logger.error("task failed %s (%s) %s%s", name, kind, context, "" if first else "; repeated")
    if first:
        _post(name, exc, context, now)
    if not unreachable:
        _record(now, subject, kind)


def _claim(key: str) -> bool:
    """True for the first claim of a key; a store that cannot answer never silences an alert."""
    url = str(settings.TASK_ALERT_REDIS_URL)
    if not url:
        return True
    try:
        store = redis.Redis.from_url(
            url, socket_connect_timeout=STORE_TIMEOUT, socket_timeout=STORE_TIMEOUT
        )
        try:
            return bool(store.set(key, "1", nx=True, ex=KEY_TTL))
        finally:
            store.close()
    except Exception as error:
        logger.error("task failure alert could not reach its store: %s", type(error).__name__)
        return True


def _post(name: str, exc: BaseException, context: str, now: datetime) -> None:
    """The class of the error only: messages of foreign errors can hold URLs and paths."""
    url = str(settings.ALERT_WEBHOOK_URL)
    if not url:
        return
    kind = type(exc).__name__
    detail = {
        "task": name,
        "error": kind,
        "message": str(exc)[:TEXT_LIMIT] if isinstance(exc, CalendarError) else None,
        "context": context,
        "first_at": now.isoformat(timespec="seconds"),
    }
    alert = {
        "date": now.date().isoformat(),
        "check": CHECK,
        "severity": DataQualityLog.Severity.WARNING.value,
        "detail": detail,
    }
    body = {"text": f"task failed: {name} ({kind})", "alerts": [alert]}
    try:
        response = requests.post(url, json=body, timeout=float(settings.ALERT_TIMEOUT))
        response.raise_for_status()
    except requests.RequestException as error:
        logger.error("task failure alert webhook failed: %s", type(error).__name__)


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
