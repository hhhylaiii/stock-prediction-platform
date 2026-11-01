"""Correct recent trading_calendar days from TWSE's traded days, e.g. after a typhoon closure."""

import logging
from datetime import date, timedelta
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from apps.etl.adapters import limiter_for
from apps.etl.adapters.twse import TwseClient
from apps.etl.errors import SourceError
from apps.etl.ratelimit import RetryableError
from apps.etl.runs import tracked_run
from apps.etl.trading_calendar import corrections, traded_days
from apps.market.models import TradingCalendar

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Correct this and last month's trading_calendar up to yesterday from TWSE FMTQIK."

    def handle(self, *args: Any, **options: Any) -> None:
        today = timezone.localdate()
        through = today - timedelta(days=1)
        start = (through.replace(day=1) - timedelta(days=1)).replace(day=1)
        rows = TradingCalendar.objects.filter(date__range=(start, through))
        stored = {
            day: (is_open, note)
            for day, is_open, note in rows.values_list("date", "is_trading_day", "note")
        }
        if not stored:
            raise CommandError(f"trading_calendar has no days in {start}..{through}")
        first = min(stored)
        try:
            with tracked_run("trading_calendar", "trading_calendar_refresh", first, through) as run:
                changed = corrections(self._traded(first, through, today), stored)
                with transaction.atomic():
                    for day, (is_open, note) in changed.items():
                        TradingCalendar.objects.filter(date=day).update(
                            is_trading_day=is_open, note=note
                        )
                run.rows = len(changed)
                run.checkpoint = {
                    "corrected": {day.isoformat(): is_open for day, (is_open, _) in changed.items()}
                }
        except (SourceError, RetryableError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        self.stdout.write(
            f"trading_calendar: {len(changed)} day(s) corrected in {first}..{through}"
        )
        for day, (is_open, _) in changed.items():
            state = "open" if is_open else "closed"
            logger.warning("trading_calendar corrected: %s is now %s", day, state)
            self.stdout.write(f"{day} is now {state}")

    @staticmethod
    def _traded(first: date, through: date, today: date) -> set[date]:
        """Traded days of every month from first to through; only this month may list none yet."""
        twse = TwseClient.build(limiter_for("twse"))
        traded: set[date] = set()
        month = first.replace(day=1)
        while month <= through:
            in_progress = (month.year, month.month) == (today.year, today.month)
            traded |= traded_days(twse.monthly_trading(month), month, allow_empty=in_progress)
            month = (month + timedelta(days=32)).replace(day=1)
        return traded
