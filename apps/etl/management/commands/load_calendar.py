"""Load trading_calendar rows from a calendar JSON file of trading days and notes."""

import json
from argparse import ArgumentParser
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Max, Min

from apps.etl.upsert import upsert
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import TradingCalendar


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
    help = "Upsert every day of a calendar file into trading_calendar."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--file", required=True)

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            calendar, notes = read_calendar_file(Path(options["file"]))
        except (OSError, KeyError, ValueError, CalendarError) as exc:
            raise CommandError(f"cannot read calendar file: {type(exc).__name__}: {exc}") from exc
        self._require_contiguous(calendar)
        frame = pd.DataFrame(
            [
                {"date": day, "is_trading_day": is_open, "note": notes.get(day.isoformat(), "")}
                for day, is_open in calendar.all_days()
            ]
        )
        rows = upsert(TradingCalendar, frame, ("date",), ("is_trading_day", "note"))
        self.stdout.write(f"trading_calendar: {rows} days {calendar.start}..{calendar.end}")

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
