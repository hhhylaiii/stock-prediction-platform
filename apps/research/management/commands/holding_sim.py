"""Read-only holding book on cached scores or stored predictions (holding-shadow-prereg.md)."""

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date, required_date
from apps.market.calendar import CalendarError, TradingDays
from apps.research.holding import (
    FULL_COSTS,
    SELL,
    SETTLE,
    STUCK,
    Book,
    Market,
    Rule,
    Settle,
    order_list,
    simulate,
)
from apps.research.holding_inputs import (
    INDEX,
    SHADOW_START,
    HoldoutRuleError,
    index_levels,
    labels_settled,
    load_market,
    research_scores,
    stored_scores,
)
from apps.research.holding_report import (
    SUMMARY_COLUMNS,
    by_year,
    index_returns,
    summarise,
    universe_returns,
)
from apps.research.labels import latest_price_date
from apps.research.prediction_cache import MissingFoldsError

RESEARCH, PREDICTION = "research", "prediction"


@dataclass(frozen=True)
class Inputs:
    scores: pd.DataFrame
    end: date
    label: str


def fold_range(text: str | None) -> list[int]:
    """8-43 as [8, ..., 43]; one number is one fold."""
    if not text:
        raise CommandError("--folds is required for the research source, e.g. 8-43")
    first, _, last = text.partition("-")
    try:
        low, high = int(first), int(last or first)
    except ValueError:
        raise CommandError(f"--folds must look like 8-43, not {text!r}") from None
    if low > high:
        raise CommandError(f"--folds {text} runs backwards")
    return list(range(low, high + 1))


def research_inputs(options: dict[str, Any], rule: Rule) -> Inputs:
    """Cached folds; research folds may not price on holdout days."""
    run_key = options["run_key"]
    if not run_key:
        raise CommandError("--run-key is required for the research source")
    folds = fold_range(options["folds"])
    if options["list"]:
        raise CommandError("--list reads stored predictions; use --source prediction")
    until = required_date(options["until"], "--until")
    end = required_date(options["end"], "--end")
    if end <= until:
        raise CommandError("--end must be after --until so the last signal can trade")
    label = f"{run_key} folds {folds[0]}-{folds[-1]}"
    return Inputs(research_scores(run_key, folds, until, end, rule), end, label)


def prediction_inputs(options: dict[str, Any], calendar: TradingDays) -> Inputs:
    """Stored rank 5-day predictions from --start, valued through the latest price day.

    The last trade day needs the day before's untradable_t1, which the evening labels settle.
    """
    start = as_date(options["start"], "--start") or SHADOW_START
    if start < SHADOW_START:
        raise CommandError(f"the shadow book starts on its first signal day {SHADOW_START}")
    latest = latest_price_date()
    if latest is None:
        raise CommandError("daily_price is empty")
    until = as_date(options["until"], "--until") or latest
    end = as_date(options["end"], "--end") or latest
    before = calendar.shift(end, -1)
    if before >= start and not labels_settled(before):
        raise CommandError(
            f"untradable_t1 of {before} is not settled until the labels are rebuilt on "
            f"{end}'s prices; pass --end {before} or run again after the evening labels"
        )
    scores = stored_scores(start, until)
    return Inputs(scores, end, f"prediction {start}..{until}")


def counts(book: Book) -> str:
    actions = book.actions["action"].value_counts()
    sells = book.actions[book.actions["action"] == SELL]
    return (
        f"sells {actions.get(SELL, 0)}, settled {actions.get(SETTLE, 0)}, "
        f"stuck days {actions.get(STUCK, 0)}, sells opening at limit down "
        f"{int(sells['limit_down_open'].sum())}"
    )


def last_list(market: Market, book: Book, rule: Rule) -> pd.DataFrame:
    """The list of the last day, whose trades the prices do not reach yet."""
    last = len(market.days) - 1
    return order_list(
        book.held,
        market.score[last],
        market.in_universe[last],
        market.stocks,
        rule,
    )


class Command(BaseCommand):
    help = (
        "Read-only: monthly net return, cost, excess over the costless equal-weight universe, "
        "its Newey-West t, TAIEX_TR, growth, drawdown and turnover of one H<x>/N<n> book."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--source", choices=[RESEARCH, PREDICTION], default=RESEARCH)
        parser.add_argument("--run-key", help="research: the cached run, e.g. rank-cand60")
        parser.add_argument("--folds", help="research: first-last fold, e.g. 8-43")
        parser.add_argument("--start", help="prediction: the first signal day")
        parser.add_argument("--until", help="the last signal day")
        parser.add_argument("--end", help="the last day valued")
        parser.add_argument("--rule", default="H70", help="H70 leaves below the 0.70 percentile")
        parser.add_argument("--size", type=int, default=30, help="slots of the book")
        parser.add_argument("--settle", choices=[item.value for item in Settle], default="live")
        parser.add_argument("--by-year", action="store_true")
        parser.add_argument("--list", action="store_true", help="print the last day's list too")

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            rule = Rule.parse(options["rule"], options["size"])
            calendar = TradingDays.from_db()
            if options["source"] == PREDICTION:
                chosen = prediction_inputs(options, calendar)
            else:
                chosen = research_inputs(options, rule)
            market = load_market(chosen.scores, chosen.end, calendar)
        except (ValueError, CalendarError, MissingFoldsError, HoldoutRuleError) as error:
            raise CommandError(str(error)) from error
        settle = Settle(options["settle"])
        book = simulate(market, rule, FULL_COSTS, settle)
        self._report(market, book, rule, settle, chosen.label, options)

    def _report(
        self,
        market: Market,
        book: Book,
        rule: Rule,
        settle: Settle,
        label: str,
        options: dict[str, Any],
    ) -> None:
        days = market.days
        self.stdout.write(f"{rule.name} {settle.value}, {label}, valued {days[0]}..{days[-1]}")
        if len(book.daily) >= 2:
            self._summary(market, book, rule, options["by_year"])
        elif not options["list"]:
            raise CommandError("the book never traded; there is nothing to summarise")
        else:
            self.stdout.write("no trade yet; the first signal day is the last day")
        if options["list"]:
            listed = last_list(market, book, rule)
            if listed.empty:
                self.stdout.write(f"no prediction stored for {days[-1]}; the book holds still")
            else:
                self.stdout.write(f"list of {days[-1]}:")
                self.stdout.write(listed.to_string(index=False))

    def _summary(self, market: Market, book: Book, rule: Rule, years: bool) -> None:
        pd.set_option("display.width", 250)
        days = market.days
        levels = index_levels(days[0], days[-1])
        absent = len(set(days) - set(levels.index))
        if absent:
            self.stdout.write(f"{INDEX} has no close on {absent} of {len(days)} days; carried over")
        universe = universe_returns(market)
        index = index_returns(levels, days)
        row = summarise(book.daily, days, universe, index)
        table = pd.DataFrame([row], index=[rule.name], columns=SUMMARY_COLUMNS).round(2)
        self.stdout.write(table.to_string())
        self.stdout.write(counts(book))
        if years:
            self.stdout.write(by_year(book.daily, days, universe, index).round(2).to_string())
