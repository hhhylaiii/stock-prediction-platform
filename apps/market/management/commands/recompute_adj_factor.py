"""Recompute daily_price.adj_factor from dividend_event, for chosen stocks or all of them."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand

from apps.market.adjust import refresh_adj_factor
from apps.market.models import DailyPrice


class Command(BaseCommand):
    help = "Rewrite adj_factor as the product of later ex-date factors (SPEC §8.2)."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--stock", action="append", dest="stocks")

    def handle(self, *args: Any, **options: Any) -> None:
        stocks = options["stocks"] or sorted(
            DailyPrice.objects.values_list("stock_id", flat=True).distinct()
        )
        changed = 0
        for stock_id in stocks:
            changed += refresh_adj_factor(stock_id)
        self.stdout.write(f"adj_factor: {changed} rows changed across {len(stocks)} stocks")
        if options["stocks"]:
            self.stdout.write(f"stocks: {', '.join(stocks)}")
