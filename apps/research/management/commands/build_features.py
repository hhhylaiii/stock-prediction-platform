"""Rebuild feature_daily and its Parquet for a range, a day, or the recent days (SPEC §6)."""

from argparse import ArgumentParser
from datetime import date, timedelta
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.etl.alerts import dispatch
from apps.etl.models import DataQualityLog
from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.research.features.build import build_feature_frame
from apps.research.features.dividend import past_calendar
from apps.research.features.store import export_year, write_features
from apps.research.labels import latest_price_date
from apps.research.lock import ResearchBusyError, research_lock
from apps.research.ranges import years

RECENT_DAYS = 60
EX_DATE_CHECK = "monitor_ex_date_calendar"


def _reported(end: date, calendar: TradingDays) -> list[Any]:
    """What end's row and the row of the session before end already said."""
    try:
        before = calendar.latest_on_or_before(end - timedelta(days=1))
    except OutsideCalendarError:
        before = end
    rows = DataQualityLog.objects.filter(check_name=EX_DATE_CHECK, date__in=[before, end])
    return [row.detail for row in rows]


def flag_ex_dates(end: date, calendar: TradingDays) -> dict[str, Any] | None:
    """Replace end's warning about announced ex-dates past the calendar; alert when it is news."""
    beyond = past_calendar(end, calendar)
    rows = DataQualityLog.objects.filter(date=end, check_name=EX_DATE_CHECK)
    earlier = _reported(end, calendar)
    if not beyond:
        rows.delete()
        return None
    detail = {
        "count": len(beyond),
        "latest_ex_date": beyond[-1].isoformat(),
        "calendar_end": calendar.end.isoformat(),
    }
    with transaction.atomic():
        rows.delete()
        row = DataQualityLog.objects.create(
            date=end,
            check_name=EX_DATE_CHECK,
            severity=DataQualityLog.Severity.WARNING,
            detail=detail,
        )
    if detail not in earlier:
        dispatch([row])
    return detail


class Command(BaseCommand):
    help = (
        "Rebuild feature_daily year by year; without a range it redoes the last 60 trading days. "
        "Holds the research lock for the whole run and fails at once when another job holds it."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--start", type=date.fromisoformat)
        parser.add_argument("--end", type=date.fromisoformat)
        parser.add_argument("--date", type=date.fromisoformat)
        parser.add_argument("--recent", type=int, default=RECENT_DAYS)

    def _range(self, calendar: TradingDays, options: dict[str, Any]) -> tuple[date, date]:
        last = latest_price_date()
        if last is None:
            raise CommandError("daily_price is empty")
        for name in ("date", "start", "end"):
            given = options[name]
            if given is not None and not calendar.start <= given <= calendar.end:
                raise CommandError(f"--{name} {given} is outside the trading calendar")
        day = options["date"]
        if day is not None:
            if options["start"] or options["end"]:
                raise CommandError("--date cannot be combined with --start or --end")
            if not calendar.is_trading_day(day):
                raise CommandError(f"{day} is not a trading day")
            if day > last:
                raise CommandError(f"{day} is after the last price day {last}")
            return day, day
        end = min(options["end"] or last, last)
        start = options["start"] or calendar.shift(
            calendar.latest_on_or_before(end), 1 - options["recent"]
        )
        if start > end:
            raise CommandError(f"--start {start} is after the end {end}")
        return start, end

    def handle(self, *args: Any, **options: Any) -> None:
        calendar = TradingDays.from_db()
        start, end = self._range(calendar, options)
        try:
            with research_lock(f"build_features {start}..{end}"):
                self.stdout.write(self._build(calendar, start, end))
        except ResearchBusyError as error:
            raise CommandError(str(error)) from error

    def _build(self, calendar: TradingDays, start: date, end: date) -> str:
        """Rebuild start..end a year at a time; the line to print."""
        rows = 0
        for low, high in years(start, end):
            rows += write_features(build_feature_frame(calendar, low, high), low, high)
            export_year(low.year, settings.FEATURE_PARQUET_DIR, rebuilt=(low, high))
        flagged = flag_ex_dates(end, calendar)
        beyond = (
            ""
            if flagged is None
            else f"; {flagged['count']} announced ex-date(s) after the trading calendar, "
            f"see {EX_DATE_CHECK}"
        )
        return f"feature_daily: {rows} rows for {start}..{end}{beyond}"
