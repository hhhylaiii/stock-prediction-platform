"""Reapply financial report filing dates from stored doc.twse pages (plan §8 K14)."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand

from apps.etl.adapters import limiter_for
from apps.etl.adapters.mops import MopsAdapter, reapply_filings
from apps.market.calendar import TradingDays


class Command(BaseCommand):
    help = "Rebuild financial_statement announce dates from raw_mops without fetching again."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--stock", action="append", dest="stocks")

    def handle(self, *args: Any, **options: Any) -> None:
        adapter = MopsAdapter(TradingDays.from_db(), limiter_for(MopsAdapter.source))
        result = reapply_filings(adapter, options["stocks"])
        self.stdout.write(f"{result.changed} statements changed across {result.stocks} stocks")
        for code, reason in sorted(result.rejected.items()):
            self.stderr.write(f"rejected {code}: {reason}")
