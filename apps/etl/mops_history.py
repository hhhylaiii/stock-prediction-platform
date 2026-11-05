"""Months and quarters FinMind lacks, filled once from MOPS's own summary pages (plan K17)."""

from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd
from django.db import models, transaction

from apps.etl.filings import quarter_end
from apps.etl.models import DataQualityLog
from apps.etl.statement_scan import figure_differences
from apps.market.models import FinancialStatement, MonthlyRevenue

REVENUE_HISTORY = "revenue_history"
STATEMENT_HISTORY = "statement_history"
HISTORIES = frozenset({REVENUE_HISTORY, STATEMENT_HISTORY})
MISMATCH_CHECKS = {
    REVENUE_HISTORY: "revenue_history_mismatch",
    STATEMENT_HISTORY: "statement_history_mismatch",
}
IFRS_FROM = "2013-Q1"
MONEY_TOLERANCE = 1000
MONTHS = 12
QUARTERS = 4


def months_between(start: date, end: date) -> list[str]:
    """Months whose first day falls in [start, end]."""
    first = start.year * MONTHS + start.month - 1 + (start.day > 1)
    last = end.year * MONTHS + end.month - 1
    return [f"{index // MONTHS:04d}-{index % MONTHS + 1:02d}" for index in range(first, last + 1)]


def quarters_between(start: date, end: date) -> list[str]:
    """IFRS quarters whose last day falls in [start, end]."""
    found = []
    for year in range(start.year, end.year + 1):
        for season in range(1, QUARTERS + 1):
            period = f"{year}-Q{season}"
            if period >= IFRS_FROM and start <= quarter_end(period) <= end:
                found.append(period)
    return found


def insert_missing_history(
    dataset: str,
    frame: pd.DataFrame,
    dated: Callable[[str], date],
    span: tuple[date, date],
    day: date,
) -> int:
    """New periods only, visible from their conservative date; stored figures are compared."""
    model: type[models.Model] = MonthlyRevenue if dataset == REVENUE_HISTORY else FinancialStatement
    stored = _stored(model, frame)
    new: list[models.Model] = []
    differing: list[dict[str, Any]] = []
    for raw in frame.to_dict("records"):
        record = {str(name): value for name, value in raw.items()}
        rows = stored.get((record["stock_id"], record["period"]))
        if rows is None:
            new.append(model(**record, announce_date=dated(record["period"])))
            continue
        row = rows.get(record.get("fin_type", ""))
        fields = {} if row is None else _differences(dataset, row, record)
        if fields:
            differing.append(
                {"stock_id": record["stock_id"], "period": record["period"], "fields": fields}
            )
    check = MISMATCH_CHECKS[dataset]
    detail = {"span": [span[0].isoformat(), span[1].isoformat()], "rows": differing}
    with transaction.atomic():
        model._default_manager.bulk_create(new)
        DataQualityLog.objects.filter(
            date=day, check_name=check, detail__span=detail["span"]
        ).delete()
        if differing:
            DataQualityLog.objects.create(
                date=day, check_name=check, severity=DataQualityLog.Severity.INFO, detail=detail
            )
    return len(new)


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
    """Figures the pages give differently (SPEC §8.5); growth rates differ by design."""
    if dataset == STATEMENT_HISTORY:
        return figure_differences(row, record)
    if row.revenue is None or abs(row.revenue - record["revenue"]) > MONEY_TOLERANCE:
        return {"revenue": [row.revenue, record["revenue"]]}
    return {}
