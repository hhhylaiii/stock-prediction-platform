"""Rebuild market_history from the stored daily foreign-holding pages of both exchanges."""

from typing import Any

from django.core.management.base import BaseCommand

from apps.etl.market_history import rebuild_market_history


class Command(BaseCommand):
    help = "Rebuild each stock's market spans from stored exchange pages, without fetching."

    def handle(self, *args: Any, **options: Any) -> None:
        report = rebuild_market_history()
        pages = ", ".join(f"{source} {count}" for source, count in report.pages.items())
        self.stdout.write(f"{report.spans} spans for {report.stocks} stocks from pages: {pages}")
        self.stdout.write(f"moved: {', '.join(report.moved)}")
        self.stdout.write(f"days listed on both markets: {report.shared_days}")
        self.stdout.write(f"unreadable pages: {report.unreadable}")
        self.stdout.write(f"listed on both markets on every day seen: {report.unsettled}")
