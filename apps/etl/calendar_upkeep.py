"""Keeps trading_calendar ahead of today: yearly extension, reach check and change log (§8.7)."""

import logging
from collections.abc import Mapping
from datetime import date, timedelta
from enum import StrEnum
from typing import Any, NamedTuple

from django.conf import settings
from django.db import transaction
from django.db.models import Max

from apps.etl.adapters import limiter_for
from apps.etl.adapters.twse import TwseClient
from apps.etl.alerts import announce
from apps.etl.errors import SchemaChangedError, SourceError
from apps.etl.models import DataQualityLog
from apps.etl.ratelimit import RetryableError
from apps.etl.runs import tracked_run
from apps.etl.task_alerts import report_swallowed
from apps.etl.trading_calendar import (
    build_calendar,
    require_whole_year,
    schedule_days,
    schedule_year,
)
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import TradingCalendar

logger = logging.getLogger(__name__)
Severity = DataQualityLog.Severity
SOURCE = "trading_calendar"
EXTEND_DATASET = "trading_calendar_extend"
REACH = "calendar_reach"
CHANGED = "calendar_changed"
REACH_BOUNDS = (60, 40, 25)
REASON_LIMIT = 200
Flips = Mapping[date, tuple[bool, bool]]


class Extension(StrEnum):
    """How one attempt to append the next year ended."""

    NOT_NEEDED = "not_needed"
    NO_YEAR_END = "no_year_end"
    UNPUBLISHED = "unpublished"
    EXTENDED = "extended"
    REFUSED = "refused"
    FAILED = "failed"


class Attempt(NamedTuple):
    """An extension outcome and, when it was refused or failed, why."""

    outcome: Extension
    reason: str = ""


def keep_up(today: date) -> Extension | None:
    """Try to extend, then record the reach: on sessions, and daily once the calendar ran out."""
    last = TradingCalendar.objects.aggregate(last=Max("date"))["last"]
    if last is None:
        return None
    session = TradingCalendar.objects.filter(date=today, is_trading_day=True).exists()
    if today <= last and not session:
        return None
    try:
        attempt = extend(today)
    except Exception as exc:
        expected = isinstance(exc, (SourceError, RetryableError))
        logger.error("trading_calendar extension failed: %s", exc, exc_info=not expected)
        if not expected:
            report_swallowed(exc, "trading_calendar extension")
        attempt = Attempt(Extension.FAILED, str(exc) if expected else type(exc).__name__)
    record_reach(today, attempt)
    return attempt.outcome


def confirmed_ahead(today: date) -> int:
    """Sessions after today that TWSE has published; provisional days do not count."""
    days = TradingCalendar.objects.filter(date__gt=today, is_trading_day=True, is_provisional=False)
    return days.count()


def extend(today: date) -> Attempt:
    """Append the year after the last confirmed day once TWSE publishes its schedule.

    A published schedule that cannot be a whole year is refused and leaves a failed run.
    """
    end = _confirmed_end()
    if end is None or (end.month, end.day) != (12, 31):
        return Attempt(Extension.NO_YEAR_END)
    if confirmed_ahead(today) >= settings.CALENDAR_EXTEND_BELOW:
        return Attempt(Extension.NOT_NEEDED)
    year = end.year + 1
    payload = TwseClient.build(limiter_for("twse")).holiday_schedule(year)
    if schedule_year(payload) != year:
        raise SchemaChangedError(f"TWSE answered another year when asked for the {year} schedule")
    if not any(day.year == year for day in schedule_days(payload)):
        return Attempt(Extension.UNPUBLISHED)
    start = end + timedelta(days=1)
    try:
        with tracked_run(SOURCE, EXTEND_DATASET, start, date(year, 12, 31)) as run:
            require_whole_year(payload)
            calendar, notes = build_calendar((), end, [payload], start)
            run.rows = _write_year(calendar, notes, today)
        logger.info("trading_calendar extended through %s", calendar.end)
    except (SchemaChangedError, CalendarError) as exc:
        logger.error("trading_calendar extension refused: %s", exc)
        return Attempt(Extension.REFUSED, str(exc))
    return Attempt(Extension.EXTENDED)


def _confirmed_end() -> date | None:
    confirmed = TradingCalendar.objects.filter(is_provisional=False)
    last: date | None = confirmed.aggregate(last=Max("date"))["last"]
    return last


def _write_year(calendar: TradingDays, notes: Mapping[str, str], today: date) -> int:
    """New days are inserted; provisional ones follow the schedule from today on."""
    span = (calendar.start, calendar.end)
    stored = dict(
        TradingCalendar.objects.filter(date__range=span).values_list("date", "is_trading_day")
    )
    rows = [
        TradingCalendar(date=day, is_trading_day=is_open, note=notes.get(day.isoformat(), ""))
        for day, is_open in calendar.all_days()
        if day not in stored or day >= today
    ]
    with transaction.atomic():
        TradingCalendar.objects.bulk_create(
            rows,
            update_conflicts=True,
            unique_fields=["date"],
            update_fields=["is_trading_day", "note", "is_provisional"],
        )
        TradingCalendar.objects.filter(date__range=span).update(is_provisional=False)
        new = {row.date: row.is_trading_day for row in rows}
        changed = log_changes("extend", today, flips(stored, new))
    if changed is not None:
        announce([changed])
    return len(rows)


def flips(stored: Mapping[date, bool], new: Mapping[date, bool]) -> dict[date, tuple[bool, bool]]:
    """Existing days whose trading flag the new values turn over, as (was, now)."""
    return {
        day: (stored[day], is_open)
        for day, is_open in new.items()
        if day in stored and stored[day] != is_open
    }


def log_changes(by: str, today: date, changed: Flips) -> DataQualityLog | None:
    """Flips of existing days go into the day's one calendar_changed row, merged with earlier."""
    if not changed:
        return None
    with transaction.atomic():
        rows = DataQualityLog.objects.select_for_update().filter(date=today, check_name=CHANGED)
        earlier = rows.first()
        detail = _merged(earlier.detail if earlier is not None else None, by, changed)
        rows.delete()
        return DataQualityLog.objects.create(
            date=today, check_name=CHANGED, severity=Severity.WARNING, detail=detail
        )


def _merged(earlier: Mapping[str, Any] | None, by: str, changed: Flips) -> dict[str, Any]:
    days = dict(earlier["days"]) if earlier is not None else {}
    for day, (was, now) in changed.items():
        first = days.get(day.isoformat(), {"was": was})
        days[day.isoformat()] = {"was": first["was"], "now": now}
    sources = sorted({*(earlier["by"] if earlier is not None else ()), by})
    return {"by": sources, "days": dict(sorted(days.items())), "earliest": min(days)}


def reach_level(left: int) -> int:
    """0 while more than 60 confirmed sessions remain, then 1, 2 and 3 at 60, 40 and 25."""
    return sum(left <= bound for bound in REACH_BOUNDS)


def record_reach(today: date, attempt: Attempt) -> DataQualityLog:
    """Today's calendar_reach row, never an error: the research gate counts every error."""
    left = confirmed_ahead(today)
    level = reach_level(left)
    refused = attempt.outcome == Extension.REFUSED
    end = _confirmed_end()
    provisional = TradingCalendar.objects.filter(is_provisional=True).aggregate(last=Max("date"))
    detail = {
        "sessions_left": left,
        "ends": end.isoformat() if end is not None else None,
        "provisional_to": provisional["last"].isoformat() if provisional["last"] else None,
        "level": level,
        "extension": str(attempt.outcome),
        "reason": attempt.reason[:REASON_LIMIT] or None,
    }
    severity = Severity.WARNING if level or refused else Severity.INFO
    with transaction.atomic():
        seen = DataQualityLog.objects.filter(check_name=REACH, date__lte=today)
        previous = seen.order_by("-date").first()
        seen.filter(date=today).delete()
        row = DataQualityLog.objects.create(
            date=today, check_name=REACH, severity=severity, detail=detail
        )
    if attempt.outcome == Extension.EXTENDED or _is_news(level, refused, previous, today):
        announce([row])
    return row


def _is_news(level: int, refused: bool, previous: DataQualityLog | None, today: date) -> bool:
    """A new level, a refused schedule, or any day at the last level; a rerun adds only news."""
    known, was_refused = 0, False
    if previous is not None:
        known = int(previous.detail["level"])
        was_refused = previous.detail["extension"] == Extension.REFUSED
    if previous is not None and previous.date == today:
        return level != known or (refused and not was_refused)
    return refused or level == len(REACH_BOUNDS) or level != known
