"""Actual filing times replace the conservative announce dates of Phase 1, earlier only (K2)."""

from collections.abc import Iterable
from datetime import date, datetime, time
from typing import Any, NamedTuple

import pandas as pd
from django.db import transaction
from django.utils import timezone

from apps.etl.models import DataQualityLog
from apps.market.calendar import TradingDays
from apps.market.models import FinancialStatement

FILED_LATER_CHECK = "statement_filed_later"
FILING_COLUMNS = ["stock_id", "period", "fin_type", "uploaded_at", "corrected"]


class Filing(NamedTuple):
    stock_id: str
    period: str
    fin_type: str
    uploaded_at: datetime
    corrected: bool


def filing_session(uploaded_at: datetime, calendar: TradingDays, cutoff: time) -> date:
    """The session whose evening run already sees the filing: the same one if before the cutoff."""
    local = timezone.localtime(uploaded_at)
    day = local.date()
    if calendar.is_trading_day(day) and local.time() < cutoff:
        return day
    return calendar.shift(calendar.latest_on_or_before(day), 1)


def apply_statement_filings(frame: pd.DataFrame, calendar: TradingDays, cutoff: time) -> int:
    """Move announce dates earlier to the filing session; later filings are only logged."""
    filings = [Filing(*row) for row in frame[FILING_COLUMNS].itertuples(index=False)]
    earlier: list[tuple[FinancialStatement, date]] = []
    later: dict[str, list[dict[str, Any]]] = {}
    for row, filing in _matched(filings):
        session = filing_session(filing.uploaded_at, calendar, cutoff)
        if session < row.announce_date:
            earlier.append((row, session))
        elif session > row.announce_date:
            later.setdefault(row.stock_id, []).append(_later(row, filing, session))
    with transaction.atomic():
        for row, session in earlier:
            FinancialStatement.objects.filter(
                stock_id=row.stock_id, period=row.period, fin_type=row.fin_type
            ).update(announce_date=session)
        _log_later({filing.stock_id for filing in filings}, later)
    return len(earlier)


def _matched(filings: list[Filing]) -> Iterable[tuple[FinancialStatement, Filing]]:
    """Each statement with its own report type's filing, else the period's latest filing."""
    by_type = {(f.stock_id, f.period, f.fin_type): f for f in filings}
    latest: dict[tuple[str, str], Filing] = {}
    for filing in filings:
        key = (filing.stock_id, filing.period)
        if key not in latest or filing.uploaded_at > latest[key].uploaded_at:
            latest[key] = filing
    rows = FinancialStatement.objects.filter(
        stock_id__in={f.stock_id for f in filings}, period__in={f.period for f in filings}
    ).order_by("stock_id", "period", "fin_type")
    for row in rows:
        found = by_type.get((row.stock_id, row.period, row.fin_type))
        found = found or latest.get((row.stock_id, row.period))
        if found is not None:
            yield row, found


def _later(row: FinancialStatement, filing: Filing, session: date) -> dict[str, Any]:
    return {
        "period": row.period,
        "fin_type": row.fin_type,
        "announce_date": row.announce_date.isoformat(),
        "filed": session.isoformat(),
        "corrected": bool(filing.corrected),
    }


def _log_later(stock_ids: set[str], later: dict[str, list[dict[str, Any]]]) -> None:
    """One warning per stock, replaced on every rerun, listing filings after the kept date."""
    for stock_id in stock_ids:
        DataQualityLog.objects.filter(
            check_name=FILED_LATER_CHECK, detail__stock_id=stock_id
        ).delete()
        if stock_id in later:
            DataQualityLog.objects.create(
                date=timezone.localdate(),
                check_name=FILED_LATER_CHECK,
                severity=DataQualityLog.Severity.WARNING,
                detail={"stock_id": stock_id, "filed_later": later[stock_id]},
            )
