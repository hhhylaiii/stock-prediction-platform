"""Fill dividend_event announce dates from stored FinMind dividend answers (plan §4.P2)."""

from argparse import ArgumentParser
from collections import Counter
from datetime import date, time
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.etl.adapters.finmind import reapply_announce_dates
from apps.etl.dividend_dates import Redated, announcement_session, moves
from apps.market.calendar import TradingDays


class Command(BaseCommand):
    help = "Fill blank dividend_event.announce_date from raw_finmind without fetching again."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--stock", action="append", dest="stocks")
        parser.add_argument(
            "--earlier",
            action="store_true",
            help="also replace a stored date the announcement precedes; never moves one forward",
        )
        parser.add_argument("--dry-run", action="store_true", help="report without writing")

    def handle(self, *args: Any, **options: Any) -> None:
        calendar = TradingDays.from_db()
        cutoff = time.fromisoformat(settings.FILING_VISIBLE_BEFORE)
        session = announcement_session(calendar, cutoff)
        earlier, dry_run = options["earlier"], options["dry_run"]
        result = reapply_announce_dates(session, options["stocks"], earlier, dry_run)
        verb = "would be dated" if dry_run else "dated"
        self.stdout.write(f"{result.filled} events {verb} across {result.stocks} stocks")
        for code, reason in sorted(result.rejected.items()):
            self.stderr.write(f"rejected {code}: {reason}")
        if dry_run:
            self._report(result.compared, calendar, earlier)

    def _report(self, compared: list[Redated], calendar: TradingDays, earlier: bool) -> None:
        """Each change a real run would make, then how many sessions stored dates trail by."""
        for row in compared:
            if moves(row, earlier):
                held = row.stored or "blank"
                self.stdout.write(f"{row.stock_id} {row.ex_date}: {held} -> {row.announced}")
        gaps = Counter(
            _sessions(calendar, row.announced, row.stored) for row in compared if row.stored
        )
        self.stdout.write(
            f"sessions the stored date trails the announcement: {dict(sorted(gaps.items()))}"
        )


def _sessions(calendar: TradingDays, announced: date, stored: date) -> int:
    """Trading sessions from the announcement to the stored date; negative when stored is first."""
    if stored >= announced:
        return len(calendar.between(announced, stored)) - 1
    return 1 - len(calendar.between(stored, announced))
