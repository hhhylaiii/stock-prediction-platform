"""Quarterly filings from the daily MOPS scan land once, dated by the scan that first saw them."""

from datetime import date, datetime, time
from typing import Any

import pandas as pd
from django.conf import settings
from django.db import transaction

from apps.etl.filings import filing_session
from apps.etl.models import DataQualityLog
from apps.etl.statement_summary import TYPE_MISMATCH_CHECK, VALUE_COLUMNS
from apps.market.calendar import TradingDays
from apps.market.models import FinancialStatement

SCAN_COLUMNS = ["stock_id", "period", "fin_type", *VALUE_COLUMNS, "announce_date"]


def scan_session(moment: datetime, calendar: TradingDays) -> date:
    """A scan begun before the evening quality check feeds that session's research run."""
    return filing_session(moment, calendar, time.fromisoformat(settings.ETL_AT["CHECK_QUALITY"]))


def insert_new_statements(frame: pd.DataFrame, day: date) -> int:
    """First sight wins: a stored period keeps its values and date; another type is logged."""
    stored = _stored_types(frame)
    new: list[FinancialStatement] = []
    mismatched: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        types = stored.get((record["stock_id"], record["period"]))
        if types is None:
            new.append(FinancialStatement(**{str(key): value for key, value in record.items()}))
        elif record["fin_type"] not in types:
            mismatched.append(
                {
                    "stock_id": record["stock_id"],
                    "period": record["period"],
                    "stored": min(types),
                    "found": record["fin_type"],
                }
            )
    with transaction.atomic():
        FinancialStatement.objects.bulk_create(new)
        DataQualityLog.objects.filter(date=day, check_name=TYPE_MISMATCH_CHECK).delete()
        if mismatched:
            DataQualityLog.objects.create(
                date=day,
                check_name=TYPE_MISMATCH_CHECK,
                severity=DataQualityLog.Severity.WARNING,
                detail={"statements": mismatched},
            )
    return len(new)


def _stored_types(frame: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    if frame.empty:
        return {}
    rows = FinancialStatement.objects.filter(
        stock_id__in=frame["stock_id"].unique().tolist(),
        period__in=frame["period"].unique().tolist(),
    ).values_list("stock_id", "period", "fin_type")
    stored: dict[tuple[str, str], set[str]] = {}
    for stock_id, period, fin_type in rows:
        stored.setdefault((stock_id, period), set()).add(fin_type)
    return stored
