"""One company's month from MOPS's JSON API, for listed months FinMind lacks (plan K17)."""

import re
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any

import pandas as pd
from django.db.models import Min

from apps.etl.deadlines import revenue_deadline
from apps.etl.errors import SchemaChangedError
from apps.etl.mops_history import months_between
from apps.etl.revenue_scan import REVENUE_COLUMNS
from apps.market.common_stock import COMMON_STOCK_PATTERN, DEPOSITARY_INDUSTRY, DEPOSITARY_PREFIX
from apps.market.models import DailyPrice, MonthlyRevenue, StockMaster

ANSWERED = 200
NO_MATCH = (406, "查無相符資料")
NOT_PUBLIC = (500, ("公開發行公司不繼續公開發行", "上市公司已下市"))
THIS_MONTH, LAST_YEAR = "本月", "去年同期"
TWD = "新台幣"
AMOUNT = re.compile(r"^-?\d+$")
QUERY_TIME = "datetime"
ANSWERS_FROM = "2014-01"
THOUSAND = 1000
ROC_OFFSET = 1911
MONTHS = 12
Figures = tuple[int, int | None]


def parse_company_revenue(payload: Mapping[str, Any], period: str) -> Figures | None:
    """This month and the same month last year in TWD; None for a month MOPS holds nothing for.

    Amounts come in thousands; foreign issuers add a column in their own currency after the TWD
    one, and a month never filed comes back as 406 with no result rather than as an error.
    """
    label = f"MOPS t05st10_ifrs {period}"
    if (payload.get("code"), payload.get("message")) == NO_MATCH:
        return None
    if payload.get("code") != ANSWERED:
        reply = f"{payload.get('code')}: {payload.get('message')}"
        raise SchemaChangedError(f"{label} answered {reply}")
    result = payload.get("result")
    expected = f"{int(period[:4]) - ROC_OFFSET}{period[5:]}"
    if not isinstance(result, dict) or result.get("yymm") != expected:
        raise SchemaChangedError(f"{label} answered for another month")
    if not result.get("data"):
        return None
    if not _twd_first(result.get("titles")):
        raise SchemaChangedError(f"{label} has no TWD column first: {result.get('titles')}")
    rows = {row[0]: row[1] for row in result["data"] if isinstance(row, list) and len(row) > 1}
    now, last_year = _amount(rows.get(THIS_MONTH)), _amount(rows.get(LAST_YEAR))
    if now is None:
        raise SchemaChangedError(f"{label} has no amount for {THIS_MONTH}: {rows.get(THIS_MONTH)}")
    return now * THOUSAND, None if last_year is None else last_year * THOUSAND


def no_longer_public(payload: Mapping[str, Any]) -> bool:
    """MOPS's refusal for a month the company was no longer public or listed, as when delisting."""
    code, phrases = NOT_PUBLIC
    message = str(payload.get("message", ""))
    return payload.get("code") == code and any(phrase in message for phrase in phrases)


def without_query_time(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The answer less the time it was asked, so a repeat answer is stored once."""
    return {key: value for key, value in payload.items() if key != QUERY_TIME}


def listed_months(
    stock_id: str, start: date, end: date, today: date, days_after: int, scanned: str | None
) -> list[str]:
    """Settled months in [start, end] from a common stock's listing month to its delisting month.

    A stock without a listing date starts at its first close. The API holds nothing before 2014,
    and months from scanned on are the daily scan's, dated by the session that saw them.
    """
    stock = (
        StockMaster.objects.filter(stock_id=stock_id, stock_id__regex=COMMON_STOCK_PATTERN)
        .exclude(stock_id__startswith=DEPOSITARY_PREFIX.removesuffix("%"))
        .exclude(industry=DEPOSITARY_INDUSTRY)
        .values("list_date", "delist_date")
        .first()
    )
    if stock is None:
        return []
    first = stock["list_date"] or _first_close(stock_id)
    if first is None:
        return []
    last = min(end, stock["delist_date"]) if stock["delist_date"] else end
    months = months_between(max(start, first.replace(day=1)), last)
    return [
        month
        for month in months
        if month >= ANSWERS_FROM
        and (scanned is None or month < scanned)
        and revenue_deadline(month) + timedelta(days=days_after) < today
    ]


def company_months(
    stock_id: str, start: date, end: date, today: date, days_after: int, scanned: str | None
) -> list[str]:
    """Listed months monthly_revenue lacks."""
    months = listed_months(stock_id, start, end, today, days_after, scanned)
    stored = set(
        MonthlyRevenue.objects.filter(stock_id=stock_id, period__in=months).values_list(
            "period", flat=True
        )
    )
    return [month for month in months if month not in stored]


def company_frame(stock_id: str, found: Mapping[str, Figures]) -> pd.DataFrame:
    """Rows in TWD; growth only against MOPS's own figures, so sources are never mixed."""
    records = [
        {
            "stock_id": stock_id,
            "period": month,
            "revenue": now,
            "yoy": _growth(now, last_year),
            "mom": _growth(now, found[_previous(month)][0] if _previous(month) in found else None),
        }
        for month, (now, last_year) in sorted(found.items())
    ]
    return pd.DataFrame(records, columns=REVENUE_COLUMNS, dtype=object)


def _twd_first(titles: Any) -> bool:
    if not isinstance(titles, list) or len(titles) < 2 or not isinstance(titles[1], dict):
        return False
    columns = titles[1].get("sub") or []
    return not columns or (isinstance(columns[0], dict) and columns[0].get("main") == TWD)


def _amount(text: Any) -> int | None:
    cleaned = str(text).replace(",", "").strip() if text is not None else ""
    return int(cleaned) if AMOUNT.match(cleaned) else None


def _first_close(stock_id: str) -> date | None:
    rows = DailyPrice.objects.filter(stock_id=stock_id, close__isnull=False)
    first: date | None = rows.aggregate(first=Min("date"))["first"]
    return first


def _previous(month: str) -> str:
    index = int(month[:4]) * MONTHS + int(month[5:]) - 2
    return f"{index // MONTHS:04d}-{index % MONTHS + 1:02d}"


def _growth(now: int, earlier: int | None) -> float | None:
    return now / earlier - 1 if earlier is not None and earlier > 0 else None
