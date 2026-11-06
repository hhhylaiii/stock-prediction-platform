"""FinMind margins taken over more than their own quarter are cleared (plan K18)."""

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from django.db import transaction

from apps.etl.errors import SchemaChangedError
from apps.etl.filings import quarter_end
from apps.etl.models import DataQualityLog
from apps.etl.mops_history import IFRS_FROM, STATEMENT_HISTORY, STATEMENT_VIEWER
from apps.etl.statement_overwrite import (
    NOT_KEPT,
    first_scanned,
    kept_summaries,
    latest_pages,
    summary_period,
)
from apps.etl.statement_summary import (
    GROSS_PROFIT,
    INCOME,
    OPERATING_INCOME,
    PARENT_NET_INCOME,
    REVENUE,
)
from apps.etl.statement_viewer import parse_viewer
from apps.market.models import FinancialStatement, FinType

MARGIN_CHECK = "statement_margin_cleared"
MARGINS = {"gross_margin": GROSS_PROFIT, "op_margin": OPERATING_INCOME}
CUMULATIVE = "cumulative"
UNVERIFIED = "unverified"
UNJUDGED = "unjudged"
TOLERANCE = 1e-4
Income = Mapping[str, float | None]
YearToDate = dict[tuple[str, str, str], Income]
Entry = dict[str, Any]


class Cleared(NamedTuple):
    kinds: Counter[str]
    unmatched: int
    untried: int
    unchecked: int
    unreadable: list[str]


class Verdict(NamedTuple):
    """Margins to clear, ones matching no span, ones whose own quarter cannot be tried, blind."""

    fields: dict[str, list[Any]]
    missed: list[Entry]
    untried: list[Entry]
    blind: int


def clear_cumulative_margins(stock_ids: Sequence[str] | None = None) -> Cleared:
    """Q2–Q4 margins spanning more than their quarter become NULL, logged per stock."""
    found, unreadable = year_to_date(stock_ids)
    kinds: Counter[str] = Counter()
    counts = Counter[str]()
    notes: dict[str, dict[str, list[Entry]]] = {}
    updated: list[tuple[FinancialStatement, list[str]]] = []
    for row in _judged_rows(stock_ids):
        if row.period < IFRS_FROM:
            verdict = _unverified(row)
        elif row.stock_id in unreadable:
            verdict = Verdict({}, [], [], sum(getattr(row, name) is not None for name in MARGINS))
        else:
            verdict = _judge(row, _same_basis(found, row))
        counts.update(
            unmatched=len(verdict.missed), untried=len(verdict.untried), blind=verdict.blind
        )
        key = {"period": row.period, "fin_type": row.fin_type}
        stock = notes.setdefault(row.stock_id, {"changes": [], "unmatched": [], "untried": []})
        stock["unmatched"].extend({**key, **miss} for miss in verdict.missed)
        stock["untried"].extend({**key, **miss} for miss in verdict.untried)
        if verdict.fields:
            kinds.update(change[2] for change in verdict.fields.values())
            for name in verdict.fields:
                setattr(row, name, None)
            updated.append((row, list(verdict.fields)))
            stock["changes"].append({**key, "fields": verdict.fields})
    with transaction.atomic():
        for row, names in updated:
            row.save(update_fields=names)
        for stock_id, stock in sorted(notes.items()):
            _log(stock_id, stock)
    return Cleared(
        kinds, counts["unmatched"], counts["untried"], counts["blind"], sorted(unreadable)
    )


def year_to_date(stock_ids: Sequence[str] | None = None) -> tuple[YearToDate, set[str]]:
    """Each company's income from January 1 by report kind, summaries first, then viewer pages.

    Companies with a viewer page that does not parse are returned apart and not judged.
    """
    found: YearToDate = {}
    unreadable: set[str] = set()
    pages = latest_pages(STATEMENT_HISTORY, ("report", "TYPEK", "year", "season"))
    for period in sorted({summary_period(params) for params in pages}):
        for code, income in kept_summaries(period, pages)[INCOME].items():
            if not stock_ids or code in stock_ids:
                found[(code, period, _kind(income))] = income
    asked = {"request_params__CO_ID__in": list(stock_ids)} if stock_ids else {}
    viewer = latest_pages(STATEMENT_VIEWER, ("CO_ID", "SYEAR", "SSEASON", "REPORT_ID"), **asked)
    for code, year, season, report in sorted(viewer):
        if stock_ids and code not in stock_ids:
            continue
        period = f"{year}-Q{season}"
        try:
            figures = _viewer_income(viewer[(code, year, season, report)], code, period)
        except SchemaChangedError:
            unreadable.add(code)
            continue
        if figures is not None:
            found.setdefault((code, period, _kind(figures)), figures)
    return found, unreadable


def spans(
    value: float, season: int, income: Mapping[int, Income], names: Iterable[str]
) -> list[int]:
    """Quarters back from the stored one whose year-to-date difference gives its margin."""
    names = tuple(names)
    matched = []
    for back in range(1, season + 1):
        start = season - back
        if start and start not in income:
            continue
        revenue = _since(income, season, start, REVENUE)
        part = _since(income, season, start, names)
        if revenue is None or part is None or revenue <= 0:
            continue
        if math.isclose(part / revenue, value, rel_tol=0, abs_tol=TOLERANCE):
            matched.append(back)
    return matched


def _judged_rows(stock_ids: Sequence[str] | None) -> Iterable[FinancialStatement]:
    rows = FinancialStatement.objects.exclude(period__endswith="Q1")
    scanned = first_scanned()
    if scanned is not None:
        rows = rows.filter(period__lt=scanned)
    if stock_ids:
        rows = rows.filter(stock_id__in=list(stock_ids))
    return rows.order_by("stock_id", "period", "fin_type").iterator()


def _unverified(row: FinancialStatement) -> Verdict:
    """Before IFRS there is no MOPS page: a margin beside a revenue FinMind left out is cleared."""
    if row.revenue is not None:
        return Verdict({}, [], [], 0)
    fields = {
        name: [getattr(row, name), None, UNVERIFIED]
        for name in MARGINS
        if getattr(row, name) is not None
    }
    return Verdict(fields, [], [], 0)


def _judge(row: FinancialStatement, income: Mapping[int, Income]) -> Verdict:
    season = int(row.period[-1])
    verdict = Verdict({}, [], [], 0)
    for name, names in MARGINS.items():
        value = getattr(row, name)
        if value is None:
            continue
        if season not in income or _value(income[season], REVENUE) is None:
            if row.revenue is None:
                verdict.fields[name] = [value, None, UNJUDGED]
            else:
                verdict = verdict._replace(blind=verdict.blind + 1)
            continue
        matched = spans(value, season, income, names)
        if not matched:
            noted = verdict.missed if season - 1 in income else verdict.untried
            noted.append({"field": name, "value": value})
        elif 1 not in matched:
            kind = CUMULATIVE if season in matched else f"{matched[0]}q"
            verdict.fields[name] = [value, None, kind]
    return verdict


def _same_basis(found: YearToDate, row: FinancialStatement) -> dict[int, Income]:
    """The row's year so far, only from reports of its own kind."""
    year = row.period[:4]
    income = {
        season: found.get((row.stock_id, f"{year}-Q{season}", row.fin_type))
        for season in range(1, 5)
    }
    return {season: figures for season, figures in income.items() if figures is not None}


def _kind(income: Income) -> str:
    consolidated = _value(income, PARENT_NET_INCOME) is not None
    return (FinType.CONSOLIDATED if consolidated else FinType.INDIVIDUAL).value


def _since(
    income: Mapping[int, Income], season: int, start: int, names: tuple[str, ...]
) -> float | None:
    now = _value(income[season], names)
    before = _value(income[start], names) if start else 0.0
    return None if now is None or before is None else now - before


def _value(income: Income, names: Iterable[str]) -> float | None:
    return next((income[name] for name in names if income.get(name) is not None), None)


def _viewer_income(page: str, code: str, period: str) -> Income | None:
    if page == NOT_KEPT:
        return None
    figures = parse_viewer(page, code, period)
    return figures[INCOME] if figures is not None else None


def _log(stock_id: str, stock: Mapping[str, list[Entry]]) -> None:
    """One row per stock, dated by its last quarter; values cleared by earlier runs stay listed."""
    logged = DataQualityLog.objects.filter(check_name=MARGIN_CHECK, detail__stock_id=stock_id)
    changes = _merged(
        [entry for log in logged for entry in log.detail["changes"]], stock["changes"]
    )
    logged.delete()
    entries = changes + stock["unmatched"] + stock["untried"]
    if not entries:
        return
    DataQualityLog.objects.create(
        date=quarter_end(max(entry["period"] for entry in entries)),
        check_name=MARGIN_CHECK,
        severity=DataQualityLog.Severity.INFO,
        detail={"stock_id": stock_id, **stock, "changes": changes},
    )


def _merged(kept: list[Entry], new: list[Entry]) -> list[Entry]:
    """Earlier entries win: a field cleared again keeps the value it had the first time."""
    seen = {
        (entry["period"], entry["fin_type"], name) for entry in kept for name in entry["fields"]
    }
    fresh = []
    for entry in new:
        fields = {
            name: change
            for name, change in entry["fields"].items()
            if (entry["period"], entry["fin_type"], name) not in seen
        }
        if fields:
            fresh.append({**entry, "fields": fields})
    return kept + fresh
