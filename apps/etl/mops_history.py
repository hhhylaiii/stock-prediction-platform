"""Months and quarters FinMind lacks, filled once from MOPS's own summary pages (plan K17)."""

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd
from django.db import models, transaction

from apps.etl.deadlines import revenue_deadline, statement_deadline
from apps.etl.filings import quarter_end
from apps.etl.models import DataQualityLog
from apps.etl.statement_scan import figure_differences
from apps.market.models import FinancialStatement, MonthlyRevenue

REVENUE_HISTORY = "revenue_history"
STATEMENT_HISTORY = "statement_history"
STATEMENT_VIEWER = "statement_viewer"
HISTORIES = frozenset({REVENUE_HISTORY, STATEMENT_HISTORY})
MISMATCH_CHECKS = {
    REVENUE_HISTORY: "revenue_history_mismatch",
    STATEMENT_HISTORY: "statement_history_mismatch",
    STATEMENT_VIEWER: "statement_viewer_mismatch",
}
SCOPE = ("span", "stock_id")
COMPARED = ("rows", "types")
IFRS_FROM = "2013-Q1"
REVENUE_FROM = "2013-01"
MONEY_TOLERANCE = 1000
MONTHS = 12
QUARTERS = 4


def months_between(start: date, end: date) -> list[str]:
    """Months from 2013 whose first day falls in [start, end]; older pages have another layout."""
    first = start.year * MONTHS + start.month - 1 + (start.day > 1)
    last = end.year * MONTHS + end.month - 1
    months = (f"{index // MONTHS:04d}-{index % MONTHS + 1:02d}" for index in range(first, last + 1))
    return [month for month in months if month >= REVENUE_FROM]


def quarters_between(start: date, end: date) -> list[str]:
    """IFRS quarters whose last day falls in [start, end]."""
    found = []
    for year in range(start.year, end.year + 1):
        for season in range(1, QUARTERS + 1):
            period = f"{year}-Q{season}"
            if period >= IFRS_FROM and start <= quarter_end(period) <= end:
                found.append(period)
    return found


def periods_between(dataset: str, start: date, end: date) -> list[str]:
    if dataset == REVENUE_HISTORY:
        return months_between(start, end)
    return quarters_between(start, end)


def unsettled(dataset: str, start: date, end: date, today: date, days_after: int) -> list[str]:
    """Periods the daily scan still dates by the session that first lists them."""
    deadline = revenue_deadline if dataset == REVENUE_HISTORY else statement_deadline
    periods = periods_between(dataset, start, end)
    return [p for p in periods if deadline(p) + timedelta(days=days_after) >= today]


def insert_missing_history(
    dataset: str,
    frame: pd.DataFrame,
    dated: Callable[[str], date],
    span: tuple[date, date],
    notes: dict[str, Any],
) -> int:
    """New periods only, from their conservative date; one log per span says what differs."""
    model: type[models.Model] = MonthlyRevenue if dataset == REVENUE_HISTORY else FinancialStatement
    stored = _stored(model, frame)
    new: list[models.Model] = []
    seen: set[tuple[str, str]] = set()
    detail: dict[str, Any] = {
        "span": [span[0].isoformat(), span[1].isoformat()],
        "added": [],
        "rows": [],
        "types": [],
        **notes,
    }
    for raw in frame.to_dict("records"):
        record = {str(name): value for name, value in raw.items()}
        key = {"stock_id": record["stock_id"], "period": record["period"]}
        seen.add((record["stock_id"], record["period"]))
        rows = stored.get((record["stock_id"], record["period"]))
        row = None if rows is None else rows.get(record.get("fin_type", ""))
        if rows is None:
            new.append(model(**record, announce_date=dated(record["period"])))
            detail["added"].append(key)
        elif row is None:
            detail["types"].append({**key, "stored": sorted(rows), "found": record["fin_type"]})
        elif fields := _differences(dataset, row, record):
            detail["rows"].append({**key, "fields": fields})
    _record(model, new, MISMATCH_CHECKS[dataset], detail, seen)
    return len(new)


def _record(
    model: type[models.Model],
    new: list[models.Model],
    check: str,
    found: Any,
    seen: set[tuple[str, str]],
) -> None:
    """Rows and the span's log together; a rerun keeps what it added and what it did not reread."""
    scope = {f"detail__{key}": found[key] for key in SCOPE if key in found}
    with transaction.atomic():
        model._default_manager.bulk_create(new)
        earlier = DataQualityLog.objects.filter(check_name=check, **scope)
        kept = [entry for log in earlier for entry in log.detail.get("added", [])]
        added = sorted({(entry["stock_id"], entry["period"]) for entry in [*kept, *found["added"]]})
        detail = {**found, "added": [{"stock_id": code, "period": p} for code, p in added]}
        for key in COMPARED:
            unread = [
                entry
                for log in earlier
                for entry in log.detail.get(key, [])
                if (entry["stock_id"], entry["period"]) not in seen
            ]
            detail[key] = [*unread, *found[key]]
        earlier.delete()
        if any(value for key, value in detail.items() if key not in SCOPE):
            DataQualityLog.objects.create(
                date=date.fromisoformat(detail["span"][1]),
                check_name=check,
                severity=DataQualityLog.Severity.INFO,
                detail=detail,
            )


def _stored(
    model: type[models.Model], frame: pd.DataFrame
) -> dict[tuple[str, str], dict[str, Any]]:
    if frame.empty:
        return {}
    rows: Any = model._default_manager.filter(
        stock_id__in=frame["stock_id"].unique().tolist(),
        period__in=frame["period"].unique().tolist(),
    )
    stored: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        stored.setdefault((row.stock_id, row.period), {})[getattr(row, "fin_type", "")] = row
    return stored


def _differences(dataset: str, row: Any, record: dict[str, Any]) -> dict[str, list[Any]]:
    """Figures the pages give differently (SPEC §8.5); growth rates differ by design.

    The viewer leaves out figures it could not work out, such as flows after an unfiled quarter.
    """
    if dataset == STATEMENT_VIEWER:
        found = figure_differences(row, record)
        return {name: pair for name, pair in found.items() if pair[1] is not None}
    if dataset != REVENUE_HISTORY:
        return figure_differences(row, record)
    if row.revenue is None or abs(row.revenue - record["revenue"]) > MONEY_TOLERANCE:
        return {"revenue": [row.revenue, record["revenue"]]}
    return {}
