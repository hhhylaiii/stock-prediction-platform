"""Print the Phase 1 acceptance numbers as Markdown (SPEC §9 Phase 1)."""

from argparse import ArgumentParser
from datetime import date
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand

from apps.etl.backfill_report import coverage, delisted_counts, latest_backfills, price_jumps

FIRST_DAY = date(2010, 1, 1)
NEW_LISTING_FREE_DAYS = 5
LIMIT_TOLERANCE = 0.001
JUMP_ROWS = 50
REJECTED_SHOWN = 20


class Command(BaseCommand):
    help = "Report table coverage, delisted stocks, unexplained price jumps and backfill runs."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--since", type=date.fromisoformat, default=FIRST_DAY)
        parser.add_argument("--output")

    def handle(self, *args: Any, **options: Any) -> None:
        sections = [
            _coverage(),
            _delisted(),
            _jumps(options["since"]),
            _runs(),
        ]
        report = "\n\n".join(sections) + "\n"
        self.stdout.write(report)
        if options["output"]:
            Path(options["output"]).write_text(report, encoding="utf-8")


def _coverage() -> str:
    lines = [
        "## Coverage",
        "",
        "| table | rows | stocks | first | last |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {item.table} | {item.rows} | {'-' if item.stocks is None else item.stocks} "
        f"| {item.first or '-'} | {item.last or '-'} |"
        for item in coverage()
    ]
    return "\n".join(lines)


def _delisted() -> str:
    delisted, priced = delisted_counts()
    return f"## Delisted stocks\n\n{delisted} in stock_master, {priced} with daily_price rows"


def _jumps(since: date) -> str:
    jumps = price_jumps(since, NEW_LISTING_FREE_DAYS, LIMIT_TOLERANCE)
    lines = [
        "## Price jumps without an event",
        "",
        f"{len(jumps)} adjusted daily moves since {since} beyond the price limit "
        f"(7% before 2015-06-01, 10% after) with no dividend_event that day, "
        f"skipping each stock's first {NEW_LISTING_FREE_DAYS} priced days",
    ]
    if jumps:
        lines += [
            "",
            "| stock | date | previous | adjusted | change |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {jump.stock_id} | {jump.day} | {jump.previous:.4f} | {jump.adjusted:.4f} "
            f"| {jump.change:+.2%} |"
            for jump in jumps[:JUMP_ROWS]
        ]
    return "\n".join(lines)


def _runs() -> str:
    lines = [
        "## Backfill runs",
        "",
        "| source | dataset | run | status | range | stocks | rows | rejected |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in latest_backfills():
        rejected = sorted(run.checkpoint.get("rejected", {}))
        shown = ", ".join(rejected[:REJECTED_SHOWN]) + (
            " …" if len(rejected) > REJECTED_SHOWN else ""
        )
        lines.append(
            f"| {run.source} | {run.dataset} | {run.run_id} | {run.status} "
            f"| {run.start_date}..{run.end_date} | {run.checkpoint.get('stocks', '-')} "
            f"| {run.rows} | {len(rejected)}{f' ({shown})' if rejected else ''} |"
        )
    return "\n".join(lines)
