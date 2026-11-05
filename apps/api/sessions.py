"""Trading-session windows for the API, always through the trading calendar (SPEC §8.7)."""

from datetime import date

from django.utils import timezone
from rest_framework.exceptions import APIException

from apps.market.calendar import CalendarError, OutsideCalendarError, TradingDays


class CalendarUnavailable(APIException):
    status_code = 503
    default_code = "calendar_unavailable"
    default_detail = "交易日曆是空的、不完整，或涵蓋不到這個日期。"


class CalendarOutdated(APIException):
    status_code = 503
    default_code = "calendar_outdated"
    default_detail = "今天或最新價格晚於交易日曆的最後一天，請先更新交易日曆。"


def load_calendar() -> TradingDays:
    try:
        return TradingDays.from_db()
    except CalendarError as exc:
        raise CalendarUnavailable() from exc


def latest_session(calendar: TradingDays) -> date:
    """The last session on or before today in Asia/Taipei."""
    today = timezone.localdate()
    if today > calendar.end:
        raise CalendarOutdated(f"今天 {today} 晚於交易日曆最後一天 {calendar.end}。")
    try:
        return calendar.latest_on_or_before(today)
    except OutsideCalendarError as exc:
        raise CalendarUnavailable() from exc


def trailing_sessions(calendar: TradingDays, last: date, count: int) -> tuple[date, ...]:
    """The count sessions ending on or before last; fewer when the calendar starts later."""
    if last > calendar.end:
        raise CalendarOutdated(f"價格日期 {last} 晚於交易日曆最後一天 {calendar.end}。")
    try:
        end = calendar.latest_on_or_before(last)
    except OutsideCalendarError as exc:
        raise CalendarUnavailable() from exc
    try:
        start = calendar.shift(end, 1 - count)
    except OutsideCalendarError:
        start = calendar.start
    return calendar.between(start, end)
