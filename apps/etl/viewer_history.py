"""Every settled quarter of a delisted company from the XBRL viewer, to insert or to compare."""

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd

from apps.etl.deadlines import statement_deadline
from apps.etl.filings import quarter_end
from apps.etl.mops_history import IFRS_FROM, quarters_between
from apps.etl.statement_summary import INCOME, REPORTS, REVENUE, previous_period, statement_rows
from apps.etl.statement_viewer import parse_viewer

KINDS = ("C", "A")
Figures = dict[str, dict[str, float | None]]
Page = Callable[[str, str], str]


def viewer_periods(
    start: date, end: date, today: date, days_after: int, delisted: date | None
) -> list[str]:
    """Settled IFRS quarters in the span that ended while the stock was still listed."""
    return [
        period
        for period in quarters_between(start, end)
        if statement_deadline(period) + timedelta(days=days_after) < today
        and (delisted is None or quarter_end(period) < delisted)
    ]


def viewer_frame(
    stock_id: str, periods: list[str], stored: set[str], page: Page, columns: list[str]
) -> pd.DataFrame:
    """Rows for every quarter: missing ones to insert, stored ones to compare (plan K18)."""
    missing = [period for period in periods if period not in stored]
    reports = _reports(stock_id, _needed(periods), page)
    records: list[dict[str, Any]] = []
    for period in periods:
        current = reports.get(period)
        before = reports.get(previous_period(period))
        if current is None:
            continue
        rows = statement_rows(
            _summary(stock_id, current), _summary(stock_id, before), int(period[-1])
        )
        records.extend({**row, "period": period} for row in rows)
    frame = pd.DataFrame(records, columns=columns, dtype=object)
    frame.attrs["absent"] = [period for period in missing if reports.get(period) is None]
    frame.attrs["stored_absent"] = [
        period for period in periods if period in stored and reports.get(period) is None
    ]
    frame.attrs["no_revenue"] = [
        period for period in missing if _lacks(reports.get(period), INCOME, REVENUE)
    ]
    return frame


def _needed(periods: list[str]) -> list[str]:
    """The quarter before each one too: flows, EPS and ROE are taken against it."""
    earlier = {previous_period(period) for period in periods}
    return sorted(set(periods) | {period for period in earlier if period >= IFRS_FROM})


def _lacks(figures: Figures | None, report: str, names: tuple[str, ...]) -> bool:
    return figures is not None and all(figures[report].get(name) is None for name in names)


def _reports(stock_id: str, periods: list[str], page: Page) -> dict[str, Figures | None]:
    """Consolidated or else individual reports; the kind that answered last is tried first."""
    kinds = list(KINDS)
    found: dict[str, Figures | None] = {}
    for period in periods:
        found[period] = None
        for kind in list(kinds):
            figures = parse_viewer(page(period, kind), stock_id, period)
            if figures is not None:
                found[period] = figures
                kinds = [kind, *(other for other in kinds if other != kind)]
                break
    return found


def _summary(stock_id: str, figures: Figures | None) -> dict[str, dict[str, Any]]:
    return {report: {} if figures is None else {stock_id: figures[report]} for report in REPORTS}
