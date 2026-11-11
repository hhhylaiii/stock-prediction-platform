"""Apply every stored ex-rights notice list again, in the order it was fetched (plan 2.7b)."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand

from apps.etl.dividend_notices import SOURCES, replay_notices
from apps.market.calendar import TradingDays


class Command(BaseCommand):
    help = "Read dividend_notice raw again; each list counts for the session of its fetch."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--source", action="append", dest="sources", choices=sorted(SOURCES))

    def handle(self, *args: Any, **options: Any) -> None:
        calendar = TradingDays.from_db()
        for source in options["sources"] or sorted(SOURCES):
            done = replay_notices(source, calendar)
            self.stdout.write(
                f"{source}: {done.lists} lists read, {done.rows} rows written, "
                f"{done.skipped} skipped"
            )
