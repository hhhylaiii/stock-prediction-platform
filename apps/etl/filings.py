"""Actual filing times replace the conservative announce dates of Phase 1 (plan §8 K2, K14)."""

from collections.abc import Iterable
from datetime import date, datetime, time
from typing import Any, NamedTuple

import pandas as pd
from django.db import transaction
from django.utils import timezone

from apps.etl.deadlines import statement_deadline, visible_from
from apps.etl.errors import SchemaChangedError
from apps.etl.models import DataQualityLog
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import FinancialStatement

FILED_LATE_CHECK = "statement_filed_late"
FILED_LATER_CHECK = "statement_filed_later"
RESTATED_CHECK = "statement_restated"
LOGS = {
    FILED_LATE_CHECK: DataQualityLog.Severity.INFO,
    FILED_LATER_CHECK: DataQualityLog.Severity.INFO,
    RESTATED_CHECK: DataQualityLog.Severity.INFO,
}
FILING_COLUMNS = ["stock_id", "period", "fin_type", "uploaded_at", "corrected", "restated"]
QUARTER_LAST_DAY = {1: 31, 2: 30, 3: 30, 4: 31}


class Filing(NamedTuple):
    stock_id: str
    period: str
    fin_type: str
    uploaded_at: datetime
    corrected: bool
    restated: bool = False


def filing_session(uploaded_at: datetime, calendar: TradingDays, cutoff: time) -> date:
    """The session whose evening run already sees the filing: the same one if before the cutoff."""
    local = timezone.localtime(uploaded_at)
    day = local.date()
    if calendar.is_trading_day(day) and local.time() < cutoff:
        return day
    return calendar.shift(calendar.latest_on_or_before(day), 1)


def apply_statement_filings(frame: pd.DataFrame, calendar: TradingDays, cutoff: time) -> int:
    """Each filing's own upload sets the date, a corrected one too: its upload is the correction."""
    filings = [Filing(*row) for row in frame[FILING_COLUMNS].itertuples(index=False)]
    changed: list[tuple[FinancialStatement, date]] = []
    logs: dict[str, dict[str, list[dict[str, Any]]]] = {check: {} for check in LOGS}
    for row, filing, own in _matched(filings):
        session = _session(row, filing, calendar, cutoff)
        deadline = visible_from(statement_deadline(row.period), calendar)
        if session != row.announce_date:
            changed.append((row, session))
        if filing.restated or session > deadline:
            check = _check(filing)
            entry = _entry(row, deadline, session, own)
            logs[check].setdefault(row.stock_id, []).append(entry)
    with transaction.atomic():
        for row, target in changed:
            FinancialStatement.objects.filter(
                stock_id=row.stock_id, period=row.period, fin_type=row.fin_type
            ).update(announce_date=target)
        seen = {(filing.stock_id, filing.period) for filing in filings}
        for check, entries in logs.items():
            _log(check, seen, entries)
    return len(changed)


def _session(row: FinancialStatement, filing: Filing, calendar: TradingDays, cutoff: time) -> date:
    """A report uploaded by the quarter's last day, or off the calendar, rejects the stock."""
    uploaded = timezone.localtime(filing.uploaded_at).date()
    if uploaded <= quarter_end(row.period):
        raise SchemaChangedError(
            f"{row.stock_id} {row.period} filed on {uploaded}, before the quarter ended"
        )
    try:
        return filing_session(filing.uploaded_at, calendar, cutoff)
    except CalendarError as exc:
        raise SchemaChangedError(f"{row.stock_id} {row.period} filed off the calendar") from exc


def _check(filing: Filing) -> str:
    if filing.restated:
        return RESTATED_CHECK
    return FILED_LATER_CHECK if filing.corrected else FILED_LATE_CHECK


def quarter_end(period: str) -> date:
    year, quarter = int(period[:4]), int(period[-1])
    return date(year, 3 * quarter, QUARTER_LAST_DAY[quarter])


def _matched(filings: list[Filing]) -> Iterable[tuple[FinancialStatement, Filing, bool]]:
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
        own = by_type.get((row.stock_id, row.period, row.fin_type))
        found = own or latest.get((row.stock_id, row.period))
        if found is not None:
            yield row, found, own is not None


def _entry(row: FinancialStatement, deadline: date, session: date, own: bool) -> dict[str, Any]:
    return {
        "period": row.period,
        "fin_type": row.fin_type,
        "deadline": deadline.isoformat(),
        "filed": session.isoformat(),
        "matched": "own" if own else "other",
    }


def _log(check: str, seen: set[tuple[str, str]], found: dict[str, list[dict[str, Any]]]) -> None:
    """One row per stock and check; a rerun replaces the periods it saw and keeps the others."""
    for stock_id in sorted({stock_id for stock_id, _ in seen}):
        logged = DataQualityLog.objects.filter(check_name=check, detail__stock_id=stock_id)
        kept = [
            entry
            for row in logged
            for entry in row.detail.get("filings", [])
            if (stock_id, entry["period"]) not in seen
        ]
        logged.delete()
        entries = sorted(
            [*kept, *found.get(stock_id, [])], key=lambda e: (e["period"], e["fin_type"])
        )
        if entries:
            DataQualityLog.objects.create(
                date=timezone.localdate(),
                check_name=check,
                severity=LOGS[check],
                detail={"stock_id": stock_id, "filings": entries},
            )
