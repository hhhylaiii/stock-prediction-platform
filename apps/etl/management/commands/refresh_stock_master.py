"""Update stock_master from one trading day's exchange quotes and the official delisting lists."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.etl.errors import SourceError
from apps.etl.management.options import required_date
from apps.etl.master_refresh import refresh_stock_master
from apps.etl.ratelimit import RetryableError
from apps.market.calendar import CalendarError, TradingDays


class Command(BaseCommand):
    help = "Add, move, relist and delist stock_master rows from a trading day's quotes (SPEC §5)."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--date", required=True)

    def handle(self, *args: Any, **options: Any) -> None:
        day = required_date(options["date"], "--date")
        try:
            run = refresh_stock_master(day, TradingDays.from_db())
        except (SourceError, RetryableError, CalendarError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        found = run.checkpoint
        self.stdout.write(
            f"stock_master {day}: {found['listed']} listed; added {len(found['added'])}, "
            f"moved {len(found['moved'])}, relisted {len(found['relisted'])}, "
            f"delisted {len(found['delisted'])}"
        )
        if found["vanished"]:
            self.stdout.write(
                f"gone from the quotes without a delisting: {', '.join(found['vanished'])}"
            )
