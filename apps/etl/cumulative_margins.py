"""FinMind margins taken over more than their own quarter are cleared (plan K18)."""

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from django.db import transaction
from django.utils import timezone

from apps.etl.errors import SchemaChangedError
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
from apps.etl.viewer_history import KINDS
from apps.market.models import FinancialStatement, FinType

MARGIN_CHECK = "statement_margin_cleared"
MARGINS = {"gross_margin": GROSS_PROFIT, "op_margin": OPERATING_INCOME}
CUMULATIVE = "cumulative"
TOLERANCE = 1e-4
Income = Mapping[str, float | None]
YearToDate = dict[tuple[str, str], Income]


class Cleared(NamedTuple):
    kinds: Counter[str]
    unmatched: int
    unchecked: int
    unreadable: list[str]


def clear_cumulative_margins(stock_ids: Sequence[str] | None = None) -> Cleared:
    """Q2–Q4 margins matching a span longer than their quarter become NULL, logged per stock."""
    found, unreadable = year_to_date(stock_ids)
    kinds: Counter[str] = Counter()
    unchecked = 0
    changes: dict[str, list[dict[str, Any]]] = {}
    unmatched: dict[str, list[dict[str, Any]]] = {}
    updated: list[tuple[FinancialStatement, list[str]]] = []
    for row in _judged_rows(stock_ids):
        income = _same_basis(found, row)
        fields, missed, blind = _judge(row, income)
        unchecked += blind
        key = {"period": row.period, "fin_type": row.fin_type}
        if missed:
            unmatched.setdefault(row.stock_id, []).extend({**key, **miss} for miss in missed)
        if fields:
            kinds.update(change[2] for change in fields.values())
            for name in fields:
                setattr(row, name, None)
            updated.append((row, list(fields)))
            changes.setdefault(row.stock_id, []).append({**key, "fields": fields})
    with transaction.atomic():
        for row, names in updated:
            row.save(update_fields=names)
        for stock_id in sorted(changes.keys() | unmatched.keys()):
            _log(stock_id, changes.get(stock_id, []), unmatched.get(stock_id, []))
    return Cleared(kinds, sum(map(len, unmatched.values())), unchecked, unreadable)


def year_to_date(stock_ids: Sequence[str] | None = None) -> tuple[YearToDate, list[str]]:
    """Each company's income from January 1, from the kept summaries, else its viewer pages."""
    found: YearToDate = {}
    pages = latest_pages(STATEMENT_HISTORY, ("report", "TYPEK", "year", "season"))
    for period in sorted({summary_period(params) for params in pages}):
        for code, income in kept_summaries(period, pages)[INCOME].items():
            if not stock_ids or code in stock_ids:
                found[(code, period)] = income
    asked = {"request_params__CO_ID__in": list(stock_ids)} if stock_ids else {}
    viewer = latest_pages(STATEMENT_VIEWER, ("CO_ID", "SYEAR", "SSEASON", "REPORT_ID"), **asked)
    periods: dict[str, set[str]] = {}
    for code, year, season, _ in viewer:
        if not stock_ids or code in stock_ids:
            periods.setdefault(code, set()).add(f"{year}-Q{season}")
    unreadable = []
    for code, read in sorted(periods.items()):
        try:
            incomes = {period: _viewer_income(code, period, viewer) for period in read}
        except SchemaChangedError:
            unreadable.append(code)
            continue
        for period, figures in incomes.items():
            if figures is not None:
                found.setdefault((code, period), figures)
    return found, unreadable


def spans(value: float, season: int, income: dict[int, Income], names: Iterable[str]) -> list[int]:
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
        if math.isclose(part / revenue, value, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            matched.append(back)
    return matched


def _judged_rows(stock_ids: Sequence[str] | None) -> Iterable[FinancialStatement]:
    rows = FinancialStatement.objects.filter(period__gte=IFRS_FROM).exclude(period__endswith="Q1")
    scanned = first_scanned()
    if scanned is not None:
        rows = rows.filter(period__lt=scanned)
    if stock_ids:
        rows = rows.filter(stock_id__in=list(stock_ids))
    return rows.order_by("stock_id", "period", "fin_type").iterator()


def _judge(
    row: FinancialStatement, income: dict[int, Income]
) -> tuple[dict[str, list[Any]], list[dict[str, Any]], int]:
    """Margins to clear, margins matching no span, and margins without figures to judge."""
    season = int(row.period[-1])
    fields: dict[str, list[Any]] = {}
    missed: list[dict[str, Any]] = []
    blind = 0
    for name, names in MARGINS.items():
        value = getattr(row, name)
        if value is None:
            continue
        if season not in income:
            blind += 1
            continue
        matched = spans(value, season, income, names)
        if not matched:
            missed.append({"field": name, "value": value})
        elif 1 not in matched:
            fields[name] = [value, None, CUMULATIVE if season in matched else f"{matched[0]}q"]
    return fields, missed, blind


def _same_basis(found: YearToDate, row: FinancialStatement) -> dict[int, Income]:
    """The row's year so far, only from reports of its own kind."""
    year = row.period[:4]
    income = {season: found.get((row.stock_id, f"{year}-Q{season}")) for season in range(1, 5)}
    return {
        season: figures
        for season, figures in income.items()
        if figures is not None and _kind(figures) == row.fin_type
    }


def _kind(income: Income) -> str:
    consolidated = _value(income, PARENT_NET_INCOME) is not None
    return (FinType.CONSOLIDATED if consolidated else FinType.INDIVIDUAL).value


def _since(
    income: dict[int, Income], season: int, start: int, names: tuple[str, ...]
) -> float | None:
    now = _value(income[season], names)
    before = _value(income[start], names) if start else 0.0
    return None if now is None or before is None else now - before


def _value(income: Income, names: Iterable[str]) -> float | None:
    return next((income[name] for name in names if income.get(name) is not None), None)


def _viewer_income(code: str, period: str, pages: dict[tuple[str, ...], str]) -> Income | None:
    for kind in KINDS:
        figures = parse_viewer(
            pages.get((code, period[:4], period[-1], kind), NOT_KEPT), code, period
        )
        if figures is not None:
            return figures[INCOME]
    return None


def _log(stock_id: str, changes: list[dict[str, Any]], unmatched: list[dict[str, Any]]) -> None:
    """One row per stock; values cleared by earlier runs stay listed."""
    logged = DataQualityLog.objects.filter(check_name=MARGIN_CHECK, detail__stock_id=stock_id)
    kept = [entry for log in logged for entry in log.detail["changes"]]
    logged.delete()
    DataQualityLog.objects.create(
        date=timezone.localdate(),
        check_name=MARGIN_CHECK,
        severity=DataQualityLog.Severity.INFO,
        detail={"stock_id": stock_id, "changes": kept + changes, "unmatched": unmatched},
    )
