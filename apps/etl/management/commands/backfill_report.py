"""Print the Phase 1 acceptance numbers as Markdown (SPEC §9 Phase 1)."""

from argparse import ArgumentParser
from datetime import date
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand

from apps.etl.adapters.finmind import FINMIND_DATASETS
from apps.etl.backfill_report import Jump, backfills, coverage, delisted_counts, price_jumps

FIRST_DAY = date(2010, 1, 1)
NEW_LISTING_FREE_DAYS = 5
LIMIT_TOLERANCE = 1e-6
FUND_ROWS = 50
DIVIDER = "| --- | --- | --- | --- | --- |"
JUMP_HEADER = ["| stock | date | previous | adjusted | change | event day |", DIVIDER + " --- |"]


class Command(BaseCommand):
    help = (
        "Report table coverage, delisted stocks, price jumps past the limit and backfill runs; "
        "run after load_listing_dates so emerging-market days are gone."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--since", type=date.fromisoformat, default=FIRST_DAY)
        parser.add_argument("--output")

    def handle(self, *args: Any, **options: Any) -> None:
        sections = [_coverage(), _delisted(), _jumps(options["since"]), _runs()]
        report = "\n\n".join(sections) + "\n"
        self.stdout.write(report)
        if options["output"]:
            Path(options["output"]).write_text(report, encoding="utf-8")


def _coverage() -> str:
    lines = ["## Coverage", "", "| table | rows | stocks | first | last |", DIVIDER]
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
    stocks = [jump for jump in jumps if not jump.fund]
    funds = [jump for jump in jumps if jump.fund]
    lines = [
        "## Price jumps past the limit",
        "",
        f"Adjusted daily moves since {since} beyond the price limit (7% before 2015-06-01, "
        f"10% after, plus {LIMIT_TOLERANCE:g} for float error), skipping the first "
        f"{NEW_LISTING_FREE_DAYS} "
        "priced days of stocks that started trading inside the data. On an event day the "
        "adjustment factor itself is suspect.",
        "",
        f"### Stocks: {len(stocks)}",
        *_jump_table(stocks),
        "",
        f"### Funds (codes starting 00, some without a price limit): {len(funds)}",
        *_jump_table(funds[:FUND_ROWS]),
    ]
    if len(funds) > FUND_ROWS:
        lines.append(f"\n{len(funds) - FUND_ROWS} more fund rows not listed")
    return "\n".join(lines)


def _jump_table(jumps: list[Jump]) -> list[str]:
    if not jumps:
        return []
    return [
        "",
        *JUMP_HEADER,
        *(
            f"| {jump.stock_id} | {jump.day} | {jump.previous:.4f} | {jump.adjusted:.4f} "
            f"| {jump.change:+.2%} | {'yes' if jump.event else ''} |"
            for jump in jumps
        ),
    ]


def _runs() -> str:
    found = backfills()
    lines = [
        "## Backfill runs",
        "",
        "Rows are summed over every whole-market run of the range (resumes add up); "
        "rejections are what is left after later runs over chosen stocks.",
        "",
        "| source | dataset | range | runs | latest | finished | last stock | stocks | rows "
        "| rejected |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in found:
        checkpoint = item.latest.checkpoint
        finished = item.latest.status == "success" or bool(checkpoint.get("finished"))
        lines.append(
            f"| {item.source} | {item.dataset} | {item.start}..{item.end} "
            f"| {', '.join(map(str, item.run_ids))} | {item.latest.status} "
            f"| {'yes' if finished else 'no'} | {checkpoint.get('last_stock_id') or '-'} "
            f"| {checkpoint.get('stocks', '-')} | {item.rows} | {len(item.outstanding)} |"
        )
    seen = {item.dataset for item in found if item.source == "finmind"}
    lines += [
        f"| finmind | {dataset} | - | - | not run | - | - | - | - | - |"
        for dataset in FINMIND_DATASETS
        if dataset not in seen
    ]
    rejected = [
        f"- {item.dataset} {code}: {reason}"
        for item in found
        for code, reason in sorted(item.outstanding.items())
    ]
    if rejected:
        lines += ["", "### Rejected stocks", "", *rejected]
    return "\n".join(lines)
