"""Fill industries the stock master lacks from the exchanges' quotes by industry (plan K17c)."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.etl.adapters import limiter_for
from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.adapters.twse import TwseAdapter
from apps.etl.industry_fill import fill_industries, sample_days
from apps.etl.management.options import required_date
from apps.market.calendar import TradingDays

ADAPTERS: dict[str, type[TwseAdapter] | type[TpexAdapter]] = {
    "twse": TwseAdapter,
    "tpex": TpexAdapter,
}


class Command(BaseCommand):
    help = "Give stocks without an industry the one the exchange listed them under each quarter."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--source", choices=sorted(ADAPTERS), required=True)
        parser.add_argument("--start", required=True)
        parser.add_argument("--end", required=True)

    def handle(self, *args: Any, **options: Any) -> None:
        start = required_date(options["start"], "--start")
        end = required_date(options["end"], "--end")
        if end > timezone.localdate():
            raise CommandError(f"--end {end} is after today")
        calendar = TradingDays.from_db()
        days = sample_days(calendar, start, end)
        if not days:
            raise CommandError(f"no sessions between {start} and {end}")
        adapter = ADAPTERS[options["source"]](calendar, limiter_for(options["source"]))
        run = fill_industries(adapter, days, since=start)
        self.stdout.write(f"{run.rows} industries filled from {len(days)} sampled days")
