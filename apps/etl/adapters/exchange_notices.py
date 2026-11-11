"""The exchanges' ex-rights notice tables: the coming ex-dates and what each is due to pay."""

import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any, NamedTuple

import pandas as pd

from apps.etl.adapters.corporate_actions import PER_THOUSAND, stock_value
from apps.etl.adapters.dispositions import list_rows
from apps.etl.adapters.exchange_actions import tpex_table
from apps.etl.dividend_dates import session_seen
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.exchange_rows import parse_price, stock_code
from apps.etl.roc import parse_roc_day
from apps.market.calendar import TradingDays

DIVIDEND_NOTICE = "dividend_notice"
TWSE_NOTICE_PATH = "/rwd/zh/exRight/TWT48U"
TPEX_NOTICE_PATH = "/www/zh-tw/bulletin/prePost"
TWSE_LABEL = "TWSE TWT48U"
TPEX_LABEL = "TPEx prePost"
TWSE_FIELDS = ("除權除息日期", "股票代號", "無償配股率", "現金股利")
TPEX_FIELDS = ("除權息日期", "代號", "每股無償配股率", "現金股利")
PENDING = ("待公告", "尚未公告")
TAGS = re.compile(r"<[^>]*>")
ZERO = Decimal(0)
NOTICE_FORM = {"response": "json"}
NOTICE_COLUMNS = ["stock_id", "ex_date", "cash_dividend", "stock_dividend"]


class Notice(NamedTuple):
    """One coming ex-date of a stock; an amount not announced yet reads as nothing due."""

    stock_id: str
    ex_date: date
    cash_dividend: Decimal
    stock_dividend: Decimal


def twse_notices(body: Mapping[str, Any]) -> list[Notice]:
    return _notices(list_rows(body, TWSE_LABEL, TWSE_FIELDS), TWSE_LABEL, TWSE_FIELDS)


def tpex_notices(body: Mapping[str, Any]) -> list[Notice]:
    table = tpex_table(body, TPEX_LABEL)
    return _notices(list_rows(table, TPEX_LABEL, TPEX_FIELDS), TPEX_LABEL, TPEX_FIELDS)


def one_day(start: date, end: date) -> None:
    """The list is only ever today's, so a run names one day and cannot fill a span."""
    if start != end:
        raise ValueError(f"{DIVIDEND_NOTICE} has no history: ask for one day, not {start}..{end}")


def kept_by_session(calendar: TradingDays, seen: datetime, payload: Any) -> list[Any]:
    """What tells one stored list from another: its session too, as lists repeat day after day."""
    return [session_seen(calendar, seen).isoformat(), payload]


def seen_at(notices: Iterable[Notice], seen: datetime, label: str) -> pd.DataFrame:
    """The list with the moment it was read; an empty one has not been published."""
    frame = pd.DataFrame(
        [notice._asdict() for notice in notices], columns=NOTICE_COLUMNS, dtype=object
    )
    if frame.empty:
        raise SourceNotReadyError(f"{label} lists nothing; the list is never empty")
    frame.attrs["seen"] = seen
    return frame


def _notices(
    records: list[dict[str, Any]], label: str, fields: tuple[str, str, str, str]
) -> list[Notice]:
    """One notice per stock and day; amounts of rows sharing both add up."""
    day, code, bonus, cash = fields
    found: dict[tuple[str, date], Notice] = {}
    for record in records:
        stock_id = stock_code(record[code])
        ex_date = parse_roc_day(record[day])
        where = f"{label} {stock_id} {ex_date}"
        paid = _due(record[cash], where)
        shares = stock_value(_due(record[bonus], where) * PER_THOUSAND)
        earlier = found.get((stock_id, ex_date))
        if earlier is not None:
            paid, shares = paid + earlier.cash_dividend, shares + earlier.stock_dividend
        found[stock_id, ex_date] = Notice(stock_id, ex_date, paid, shares)
    return list(found.values())


def _due(text: Any, where: str) -> Decimal:
    """A non-negative amount; one the issuer has yet to announce counts as none for now."""
    cleaned = TAGS.sub("", str(text)).strip()
    if any(mark in cleaned for mark in PENDING):
        return ZERO
    value = parse_price(cleaned)
    if value is None or value < 0:
        raise SchemaChangedError(f"{where} has an unreadable amount {text!r}")
    return value
