"""Trading-session windows for the API, always through the trading calendar (SPEC §8.7)."""

from datetime import date

from django.utils import timezone
from rest_framework.exceptions import APIException

from apps.market.calendar import CalendarError, OutsideCalendarError, TradingDays


class CalendarUnavailable(APIException):
    status_code = 503
    default_code = "calendar_unavailable"
    default_detail = "The trading calendar is empty, incomplete or does not reach this date."


class CalendarOutdated(APIException):
    status_code = 503
    default_code = "calendar_outdated"
    default_detail = "Today or the latest prices fall after the end of the trading calendar."


def load_calendar() -> TradingDays:
    try:
        return TradingDays.from_db()
    except CalendarError as exc:
        raise CalendarUnavailable(str(exc)) from exc


def latest_session(calendar: TradingDays) -> date:
    """The last session on or before today in Asia/Taipei."""
    today = timezone.localdate()
    if today > calendar.end:
        raise CalendarOutdated(f"today {today} is after the calendar end {calendar.end}")
    try:
        return calendar.latest_on_or_before(today)
    except OutsideCalendarError as exc:
        raise CalendarUnavailable(str(exc)) from exc


def trailing_sessions(calendar: TradingDays, last: date, count: int) -> tuple[date, ...]:
    """The count sessions ending on or before last; fewer when the calendar starts later."""
    if last > calendar.end:
        raise CalendarOutdated(f"{last} is after the calendar end {calendar.end}")
    try:
        end = calendar.latest_on_or_before(last)
    except OutsideCalendarError as exc:
        raise CalendarUnavailable(str(exc)) from exc
    try:
        start = calendar.shift(end, 1 - count)
    except OutsideCalendarError:
        start = calendar.start
    return calendar.between(start, end)
