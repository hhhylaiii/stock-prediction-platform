"""Fill dividend_event announce dates from stored FinMind dividend answers (plan §4.P2)."""

from argparse import ArgumentParser
from datetime import time
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.etl.adapters.finmind import reapply_announce_dates
from apps.etl.dividend_dates import announcement_session
from apps.market.calendar import TradingDays


class Command(BaseCommand):
    help = "Fill blank dividend_event.announce_date from raw_finmind without fetching again."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--stock", action="append", dest="stocks")

    def handle(self, *args: Any, **options: Any) -> None:
        cutoff = time.fromisoformat(settings.FILING_VISIBLE_BEFORE)
        session = announcement_session(TradingDays.from_db(), cutoff)
        result = reapply_announce_dates(session, options["stocks"])
        self.stdout.write(f"{result.filled} events dated across {result.stocks} stocks")
        for code, reason in sorted(result.rejected.items()):
            self.stderr.write(f"rejected {code}: {reason}")
