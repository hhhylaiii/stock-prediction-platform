"""Check sampled stock-years of daily_price against TWSE and TPEx per-stock monthly tables."""

from argparse import ArgumentParser
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any, NamedTuple

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.etl.adapters import limiter_for
from apps.etl.adapters.tpex import TpexClient
from apps.etl.adapters.twse import TwseClient
from apps.etl.exchange_rows import OfficialDay
from apps.etl.verify import Difference, compare_days, sample_pairs, stored_days
from apps.market.models import Market, StockMaster

DEFAULT_SEED = 20261101
DETAIL_LIMIT = 300
MONTHS = 12
Lookup = Callable[[str, date], list[OfficialDay] | None]


class Span(NamedTuple):
    stock_id: str
    label: str
    months: list[date]


class Checked(NamedTuple):
    span: Span
    official: int
    stored: int
    differences: list[Difference]


class Command(BaseCommand):
    help = "Compare sampled stock-years of daily_price with TWSE STOCK_DAY and TPEx tradingStock."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--sample", type=int, default=20)
        parser.add_argument("--years", type=int, default=3)
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--last-year", type=int, dest="last_year")
        parser.add_argument("--pair", action="append", dest="pairs", metavar="CODE:YEAR")
        parser.add_argument("--output")

    def handle(self, *args: Any, **options: Any) -> None:
        today = timezone.localdate()
        if options["pairs"]:
            spans = [_explicit(text, today) for text in options["pairs"]]
        else:
            last_year = options["last_year"] or today.year - 1
            pairs = sample_pairs(options["sample"], options["years"], options["seed"], last_year)
            spans = [_year_span(code, year, today) for code, year in pairs]
        if not spans:
            raise CommandError("no stock has prices to sample")
        unknown = {span.stock_id for span in spans} - set(
            StockMaster.objects.values_list("stock_id", flat=True)
        )
        if unknown:
            raise CommandError(f"not in stock_master: {', '.join(sorted(unknown))}")
        twse, tpex = TwseClient.build(limiter_for("twse")), TpexClient.build(limiter_for("tpex"))
        checked = [_check(span, twse, tpex) for span in spans]
        report = _render(checked, options["seed"])
        self.stdout.write(report)
        if options["output"]:
            Path(options["output"]).write_text(report, encoding="utf-8")
        total = sum(len(item.differences) for item in checked)
        if total:
            noun = "stock-year" if len(checked) == 1 else "stock-years"
            raise CommandError(f"{total} differences in {len(checked)} {noun}")


def _explicit(text: str, today: date) -> Span:
    """CODE:YEAR checks a whole year; CODE:YYYY-MM a single month."""
    code, _, when = text.partition(":")
    try:
        if len(when) == len("YYYY"):
            return _year_span(code, int(when), today)
        month = date.fromisoformat(f"{when}-01")
    except ValueError:
        raise CommandError("--pair takes CODE:YEAR or CODE:YYYY-MM") from None
    return Span(code, when, [month])


def _year_span(code: str, year: int, today: date) -> Span:
    months = [date(year, number, 1) for number in range(1, MONTHS + 1)]
    return Span(code, str(year), [month for month in months if month <= today])


def _check(span: Span, twse: TwseClient, tpex: TpexClient) -> Checked:
    stock = StockMaster.objects.get(stock_id=span.stock_id)
    lookups: list[Lookup] = [twse.stock_days, tpex.stock_days]
    if stock.market == Market.TPEX:
        lookups.reverse()
    last_day = _next_month(span.months[-1]) - timedelta(days=1)
    stored = stored_days(span.stock_id, span.months[0], last_day)
    official: dict[date, OfficialDay] = {}
    for month in span.months:
        wanted = {day for day in stored if day.replace(day=1) == month}
        for lookup in lookups:
            days = lookup(span.stock_id, month) or []
            official |= {day.day: day for day in days}
            if days and wanted <= official.keys():
                break
    differences = compare_days(span.stock_id, official, stored)
    return Checked(span, len(official), len(stored), differences)


def _next_month(month: date) -> date:
    return date(month.year + month.month // MONTHS, month.month % MONTHS + 1, 1)


def _render(checked: list[Checked], seed: int) -> str:
    total = sum(len(item.differences) for item in checked)
    lines = [
        f"verify_backfill: {len(checked)} stock-years, {total} differences (seed {seed})",
        "",
        "| stock | span | official days | stored days | differences |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {item.span.stock_id} | {item.span.label} | {item.official} | {item.stored} "
        f"| {len(item.differences)} |"
        for item in checked
    ]
    details = [difference for item in checked for difference in item.differences]
    if details:
        lines += [
            "",
            "| stock | date | field | stored | official |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {d.stock_id} | {d.day} | {d.field} | {d.stored} | {d.official} |"
            for d in details[:DETAIL_LIMIT]
        ]
        if len(details) > DETAIL_LIMIT:
            lines.append(f"\n{len(details) - DETAIL_LIMIT} more differences not listed")
    return "\n".join(lines) + "\n"
