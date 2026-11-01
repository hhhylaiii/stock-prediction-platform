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
from apps.etl.errors import SourceError
from apps.etl.exchange_rows import OfficialDay
from apps.etl.ratelimit import RetryableError
from apps.etl.verify import FIRST_YEAR, Difference, compare_days, sample_pairs, stored_days
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import Market, StockMaster

DEFAULT_SEED = 20261101
DETAIL_LIMIT = 300
MONTHS = 12
DIVIDER = "| --- | --- | --- | --- | --- |"
TPEX_NOTE = (
    "TPEx volumes arrive in thousands of shares rounded to the nearest thousand and leave out "
    "block trades (鉅額交易), so a TPEx volume difference on a block-trade day may not be an error."
)
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
    error: str = ""


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
        spans, chosen = self._spans(options, today)
        try:
            calendar = TradingDays.from_db()
        except CalendarError as exc:
            raise CommandError(f"CalendarError: {exc}") from exc
        twse, tpex = TwseClient.build(limiter_for("twse")), TpexClient.build(limiter_for("tpex"))
        checked = [_checked(span, calendar, twse, tpex) for span in spans]
        report = _render(checked, chosen)
        self.stdout.write(report)
        if options["output"]:
            Path(options["output"]).write_text(report, encoding="utf-8")
        total = sum(len(item.differences) for item in checked)
        failed = sum(1 for item in checked if item.error)
        if total or failed:
            noun = "stock-year" if len(checked) == 1 else "stock-years"
            raise CommandError(f"{total} differences, {failed} failed in {len(checked)} {noun}")

    def _spans(self, options: dict[str, Any], today: date) -> tuple[list[Span], str]:
        if options["pairs"]:
            spans = [_explicit(text, today) for text in options["pairs"]]
            chosen = "pairs given on the command line"
        else:
            last_year = options["last_year"] or today.year - 1
            pairs = sample_pairs(options["sample"], options["years"], options["seed"], last_year)
            spans = [_year_span(code, year, today) for code, year in pairs]
            chosen = (
                f"{options['sample']} stocks x {options['years']} years, seed {options['seed']}, "
                f"years {FIRST_YEAR}..{last_year}"
            )
        if not spans:
            raise CommandError("no stock has prices to sample")
        known = set(StockMaster.objects.values_list("stock_id", flat=True))
        unknown = {span.stock_id for span in spans} - known
        if unknown:
            raise CommandError(f"not in stock_master: {', '.join(sorted(unknown))}")
        return spans, chosen


def _explicit(text: str, today: date) -> Span:
    """CODE:YEAR checks a whole year; CODE:YYYY-MM a single month."""
    code, _, when = text.partition(":")
    try:
        if len(when) == len("YYYY"):
            return _year_span(code, int(when), today)
        month = date.fromisoformat(f"{when}-01")
    except ValueError:
        raise CommandError("--pair takes CODE:YEAR or CODE:YYYY-MM") from None
    if month > today:
        raise CommandError(f"{text} is in the future")
    return Span(code, when, [month])


def _year_span(code: str, year: int, today: date) -> Span:
    months = [date(year, number, 1) for number in range(1, MONTHS + 1)]
    past = [month for month in months if month <= today]
    if not past:
        raise CommandError(f"{code}:{year} is in the future")
    return Span(code, str(year), past)


def _checked(span: Span, calendar: TradingDays, twse: TwseClient, tpex: TpexClient) -> Checked:
    """A failed lookup marks the span and lets the others finish."""
    try:
        return _check(span, calendar, twse, tpex)
    except (SourceError, RetryableError, CalendarError) as exc:
        return Checked(span, 0, 0, [], f"{type(exc).__name__}: {exc}")


def _check(span: Span, calendar: TradingDays, twse: TwseClient, tpex: TpexClient) -> Checked:
    stock = StockMaster.objects.get(stock_id=span.stock_id)
    lookups: list[Lookup] = [twse.stock_days, tpex.stock_days]
    if stock.market == Market.TPEX:
        lookups.reverse()
    last_day = _next_month(span.months[-1]) - timedelta(days=1)
    stored = stored_days(span.stock_id, span.months[0], last_day)
    official: dict[date, OfficialDay] = {}
    unanswered = []
    for month in span.months:
        month_end = _next_month(month) - timedelta(days=1)
        needed = set(calendar.between(month, month_end)) | {
            day for day in stored if month <= day <= month_end
        }
        answers = []
        for lookup in lookups:
            days = lookup(span.stock_id, month)
            answers.append(days)
            official |= {day.day: day for day in days or []}
            if needed <= official.keys():
                break
        if None in answers and not needed <= official.keys():
            unanswered.append(f"{month:%Y-%m}")
    differences = compare_days(span.stock_id, official, stored)
    error = f"a market cannot answer {', '.join(unanswered)}" if unanswered else ""
    return Checked(span, len(official), len(stored), differences, error)


def _next_month(month: date) -> date:
    return date(month.year + month.month // MONTHS, month.month % MONTHS + 1, 1)


def _render(checked: list[Checked], chosen: str) -> str:
    total = sum(len(item.differences) for item in checked)
    lines = [
        f"verify_backfill: {len(checked)} stock-years, {total} differences ({chosen})",
        "",
        TPEX_NOTE,
        "",
        "| stock | span | official days | stored days | differences | error |",
        DIVIDER + " --- |",
    ]
    lines += [
        f"| {item.span.stock_id} | {item.span.label} | {item.official} | {item.stored} "
        f"| {len(item.differences)} | {item.error} |"
        for item in checked
    ]
    details = [difference for item in checked for difference in item.differences]
    if details:
        lines += ["", "| stock | date | field | stored | official |", DIVIDER]
        lines += [
            f"| {d.stock_id} | {d.day} | {d.field} | {d.stored} | {d.official} |"
            for d in details[:DETAIL_LIMIT]
        ]
        if len(details) > DETAIL_LIMIT:
            lines.append(f"\n{len(details) - DETAIL_LIMIT} more differences not listed")
    return "\n".join(lines) + "\n"
