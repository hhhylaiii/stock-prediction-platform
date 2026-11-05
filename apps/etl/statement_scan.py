"""Quarterly filings from the daily MOPS scan land once, dated by the scan that first saw them."""

from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

import pandas as pd
from django.conf import settings
from django.db import transaction

from apps.etl.filings import filing_session
from apps.etl.models import DataQualityLog
from apps.etl.quality import replace_day_log
from apps.etl.statement_summary import TYPE_MISMATCH_CHECK, VALUE_COLUMNS
from apps.market.calendar import TradingDays
from apps.market.models import FinancialStatement

SCAN_COLUMNS = ["stock_id", "period", "fin_type", *VALUE_COLUMNS]
SOURCE_MISMATCH_CHECK = "statement_source_mismatch"
MONEY_TOLERANCE = 1000
EPS_TOLERANCE = 0.02
RATIO_TOLERANCE = 1e-6
TOLERANCES = {
    "eps": EPS_TOLERANCE,
    "revenue": MONEY_TOLERANCE,
    "net_income": MONEY_TOLERANCE,
    "operating_cf": MONEY_TOLERANCE,
}
Severity = DataQualityLog.Severity


def scan_session(moment: datetime, calendar: TradingDays) -> date:
    """Rows written before the evening quality check count for that session."""
    return filing_session(moment, calendar, time.fromisoformat(settings.ETL_AT["CHECK_QUALITY"]))


def insert_new_statements(frame: pd.DataFrame, day: date, session: date) -> int:
    """First sight wins; a stored period keeps its values and date, and differences are logged."""
    stored = _stored(frame)
    new: list[FinancialStatement] = []
    other_type: list[dict[str, Any]] = []
    other_figures: list[dict[str, Any]] = []
    for raw in frame.to_dict("records"):
        record = {str(name): value for name, value in raw.items()}
        key = {"stock_id": record["stock_id"], "period": record["period"]}
        rows = stored.get((record["stock_id"], record["period"]))
        if rows is None:
            new.append(FinancialStatement(**record, announce_date=session))
        elif record["fin_type"] not in rows:
            other_type.append({**key, "stored": min(rows), "found": record["fin_type"]})
        else:
            fields = _differences(rows[record["fin_type"]], record)
            if fields:
                other_figures.append({**key, "fin_type": record["fin_type"], "fields": fields})
    with transaction.atomic():
        FinancialStatement.objects.bulk_create(new)
        replace_day_log(day, TYPE_MISMATCH_CHECK, Severity.WARNING, _detail(other_type))
        replace_day_log(day, SOURCE_MISMATCH_CHECK, Severity.INFO, _detail(other_figures))
    return len(new)


def _stored(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, FinancialStatement]]:
    if frame.empty:
        return {}
    rows = FinancialStatement.objects.filter(
        stock_id__in=frame["stock_id"].unique().tolist(),
        period__in=frame["period"].unique().tolist(),
    )
    stored: dict[tuple[str, str], dict[str, FinancialStatement]] = {}
    for row in rows:
        stored.setdefault((row.stock_id, row.period), {})[row.fin_type] = row
    return stored


def _differences(row: FinancialStatement, record: dict[str, Any]) -> dict[str, list[Any]]:
    """Figures the summary gives differently from the stored row (SPEC §8.5)."""
    fields = {}
    for name in VALUE_COLUMNS:
        stored, found = getattr(row, name), record[name]
        if stored is None and found is None:
            continue
        tolerance = TOLERANCES.get(name, RATIO_TOLERANCE)
        if stored is None or found is None or abs(float(stored) - float(found)) > tolerance:
            fields[name] = [_plain(stored), _plain(found)]
    return fields


def _plain(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


def _detail(statements: list[dict[str, Any]]) -> dict[str, Any] | None:
    return {"statements": statements} if statements else None
