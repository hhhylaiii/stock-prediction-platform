"""Show what today's ex-rights notice lists would write, keeping none of it (plan 2.7b)."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from apps.etl.adapters import build_adapter
from apps.etl.adapters.exchange_notices import DIVIDEND_NOTICE
from apps.etl.dividend_notices import SOURCES, Outcome, preview_notices
from apps.etl.errors import SourceError
from apps.etl.ratelimit import RetryableError
from apps.market.calendar import CalendarError, TradingDays


class Command(BaseCommand):
    help = "Fetch the dividend_notice lists and print what a real run would write; nothing stays."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--source", action="append", dest="sources", choices=sorted(SOURCES))

    def handle(self, *args: Any, **options: Any) -> None:
        calendar = TradingDays.from_db()
        try:
            day = calendar.latest_on_or_before(timezone.localdate())
            for source in options["sources"] or sorted(SOURCES):
                self._report(source, _previewed(source, calendar, day))
        except (ValueError, CalendarError, SourceError, RetryableError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        self.stdout.write("nothing was written")

    def _report(self, source: str, outcome: Outcome) -> None:
        detail = outcome.detail
        state = "sound" if detail["sound"] else "NOT sound"
        self.stdout.write(
            f"{source} session {outcome.session}: {detail['listed']} listed, {state}, "
            f"{len(outcome.created)} to create, {len(outcome.dated)} to date, "
            f"{len(outcome.revised)} to revise, {len(detail['removed_now'])} to remove"
        )
        for row in outcome.created:
            self.stdout.write(f"  create {_event(row)} announced {row['announce_date']}")
        for row in outcome.dated:
            self.stdout.write(
                f"  date {row['stock_id']} {row['ex_date']} announced {row['announce_date']}"
            )
        for row in outcome.revised:
            self.stdout.write(
                f"  revise {_event(row)} was cash {row['was'][0]} stock {row['was'][1]}"
            )
        for row in detail["removed_now"]:
            self.stdout.write(f"  remove {_event(row)} announced {row['announce_date']}")
        for name in ("gone", "unsettled"):
            if detail[name]:
                self.stdout.write(f"  {name}: {', '.join(detail[name])}")


def _previewed(source: str, calendar: TradingDays, day: Any) -> Outcome:
    """The same fetch and write as a real run, inside a transaction that is always undone."""
    adapter = build_adapter(source, calendar)
    with transaction.atomic():
        frame = adapter.fetch(DIVIDEND_NOTICE, day, day)
        outcome = preview_notices(source, frame, calendar)
        transaction.set_rollback(True)
    return outcome


def _event(row: dict[str, Any]) -> str:
    return (
        f"{row['stock_id']} {row['ex_date']} cash {row['cash_dividend']} "
        f"stock {row['stock_dividend']}"
    )
