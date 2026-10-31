"""Load trading_calendar from a calendar file, or rebuild it from TWSE (SPEC §5, §8.7)."""

import json
from argparse import ArgumentParser
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Max, Min
from django.utils import timezone

from apps.etl.adapters import limiter_for
from apps.etl.adapters.twse import TwseClient
from apps.etl.errors import SourceError
from apps.etl.management.options import required_date
from apps.etl.ratelimit import RetryableError
from apps.etl.runs import tracked_run
from apps.etl.trading_calendar import build_calendar, traded_days
from apps.etl.upsert import upsert
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import TradingCalendar

SCHEDULE_YEARS_AHEAD = 1


def read_calendar_file(path: Path) -> tuple[TradingDays, dict[str, str]]:
    """Parse {start, end, trading_days, notes}; every trading day must fall inside the range."""
    data = json.loads(path.read_text(encoding="utf-8"))
    calendar = TradingDays(
        (date.fromisoformat(day) for day in data["trading_days"]),
        date.fromisoformat(data["start"]),
        date.fromisoformat(data["end"]),
    )
    notes = {str(key): str(value) for key, value in data.get("notes", {}).items()}
    return calendar, notes


class Command(BaseCommand):
    help = "Upsert trading_calendar from a calendar file or from TWSE trading days and schedule."

    def add_arguments(self, parser: ArgumentParser) -> None:
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--file")
        group.add_argument("--from", dest="start")

    def handle(self, *args: Any, **options: Any) -> None:
        if options["file"] is not None:
            calendar, notes = self._from_file(Path(options["file"]))
            self._write(calendar, notes)
            return
        start = required_date(options["start"], "--from")
        today = timezone.localdate()
        if start >= today:
            raise CommandError(f"--from {start} must be before today ({today})")
        try:
            with tracked_run("trading_calendar", "trading_calendar", start, today) as run:
                calendar, notes = self._from_sources(start, today)
                run.end_date = calendar.end
                run.rows = self._write(calendar, notes)
        except (SourceError, RetryableError, CalendarError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _from_file(path: Path) -> tuple[TradingDays, dict[str, str]]:
        try:
            return read_calendar_file(path)
        except (OSError, KeyError, TypeError, ValueError, CalendarError) as exc:
            raise CommandError(f"cannot read calendar file: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _from_sources(start: date, today: date) -> tuple[TradingDays, dict[str, str]]:
        """Traded days through yesterday from FMTQIK; today onwards from the holiday schedule."""
        twse = TwseClient.build(limiter_for("twse"))
        yesterday = today - timedelta(days=1)
        current = today.replace(day=1)
        traded: set[date] = set()
        month = start.replace(day=1)
        while month <= yesterday:
            payload = twse.monthly_trading(month)
            traded |= traded_days(payload, month, allow_empty=month == current)
            month = (month + timedelta(days=32)).replace(day=1)
        schedules = []
        for year in range(today.year, today.year + SCHEDULE_YEARS_AHEAD + 1):
            schedules.append(twse.holiday_schedule(year))
            if not schedules[-1].get("data"):
                break
        return build_calendar(traded, yesterday, schedules, start)

    def _write(self, calendar: TradingDays, notes: dict[str, str]) -> int:
        self._require_contiguous(calendar)
        frame = pd.DataFrame(
            [
                {"date": day, "is_trading_day": is_open, "note": notes.get(day.isoformat(), "")}
                for day, is_open in calendar.all_days()
            ]
        )
        rows = upsert(TradingCalendar, frame, ("date",), ("is_trading_day", "note"))
        self.stdout.write(f"trading_calendar: {rows} days {calendar.start}..{calendar.end}")
        return rows

    @staticmethod
    def _require_contiguous(calendar: TradingDays) -> None:
        """The table must stay gap-free, so a new file has to overlap or touch what is loaded."""
        bounds = TradingCalendar.objects.aggregate(first=Min("date"), last=Max("date"))
        first, last = bounds["first"], bounds["last"]
        if first is None or last is None:
            return
        if calendar.start > last + timedelta(days=1) or calendar.end < first - timedelta(days=1):
            raise CommandError(
                f"{calendar.start}..{calendar.end} would leave a gap next to the loaded "
                f"{first}..{last}; load the days in between first"
            )
