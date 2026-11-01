"""TWSE whole-market daily tables mapped onto standard rows (SPEC §4, §5)."""

from datetime import date
from typing import Any

import pandas as pd

from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.exchange_rows import parse_count, parse_price
from apps.etl.roc import parse_roc_date
from apps.market.models import IndexId

NO_DATA_STAT = "沒有符合條件的資料"
AFTER_LATEST_STAT = "查詢日期大於可查詢最大日期"
FOREIGN_CURRENCY_CODE_SUFFIXES = frozenset("KMSC")
MONTHLY_UNIT_HINT = "單位：元、股"
MONTHLY_FIELDS = ("日期", "成交股數", "成交金額", "成交筆數", "發行量加權股價指數", "漲跌點數")
INDEX_COLUMNS = ["index_id", "date", "close", "volume"]


def is_foreign_currency_etf(code: str) -> bool:
    """TWSE: an ETF code whose sixth character is K, M, S or C trades in a foreign currency."""
    return len(code) == 6 and code.startswith("00") and code[5] in FOREIGN_CURRENCY_CODE_SUFFIXES


def find_table(payload: dict[str, Any], title_part: str) -> dict[str, Any]:
    for table in payload.get("tables", []):
        if title_part in (table.get("title") or ""):
            return dict(table)
    raise SchemaChangedError(f"no table titled *{title_part}* in TWSE response")


def check_answer(payload: dict[str, Any], day: date) -> None:
    """The payload must be a successful answer for the requested day itself."""
    stat = str(payload.get("stat", ""))
    if stat != "OK":
        raise SchemaChangedError(f"TWSE answered with unexpected stat {stat!r}")
    requested = day.strftime("%Y%m%d")
    answered = str(payload.get("date", ""))
    if answered == requested:
        return
    if len(answered) == len(requested) and answered.isdigit() and answered < requested:
        raise SourceNotReadyError(f"TWSE still serves {answered} when asked for {day}")
    raise SchemaChangedError(f"TWSE answered for {answered!r} instead of {requested}")


def check_published(payload: dict[str, Any], label: str, day: date) -> None:
    """TWSE says 'no data', 'after the latest date' or sends only empty tables before publishing."""
    stat = str(payload.get("stat", ""))
    if NO_DATA_STAT in stat or AFTER_LATEST_STAT in stat:
        raise SourceNotReadyError(f"TWSE {label} for {day} not available: {stat}")
    bodies = payload.get("tables", [payload])
    if stat == "OK" and not any(isinstance(body, dict) and body.get("data") for body in bodies):
        raise SourceNotReadyError(f"TWSE {label} for {day} has no rows yet")


def taiex_rows(payload: dict[str, Any], day: date) -> pd.DataFrame:
    """The FMTQIK row of the day; a day after the last row is not published yet."""
    if str(payload.get("stat", "")) != "OK":
        raise SchemaChangedError(f"TWSE FMTQIK answered with stat {payload.get('stat')!r}")
    if payload.get("hints") != MONTHLY_UNIT_HINT:
        raise SchemaChangedError(f"TWSE FMTQIK units changed: {payload.get('hints')!r}")
    if tuple(payload.get("fields") or ()) != MONTHLY_FIELDS:
        raise SchemaChangedError(f"TWSE FMTQIK fields changed: {payload.get('fields')}")
    rows = payload.get("data")
    if not isinstance(rows, list) or not all(
        isinstance(row, list) and len(row) == len(MONTHLY_FIELDS) for row in rows
    ):
        raise SchemaChangedError("TWSE FMTQIK rows are malformed")
    by_day = {parse_roc_date(str(row[0])): row for row in rows}
    if any(listed.replace(day=1) != day.replace(day=1) for listed in by_day):
        raise SchemaChangedError(f"TWSE FMTQIK answered {day:%Y-%m} with days of another month")
    if day in by_day:
        return pd.DataFrame([by_day[day]], columns=list(MONTHLY_FIELDS)).assign(date=day)
    if not by_day or day > max(by_day):
        raise SourceNotReadyError(f"TWSE FMTQIK has no row for {day} yet")
    raise SchemaChangedError(f"TWSE FMTQIK skips trading day {day}; the calendar may be wrong")


def taiex_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in raw_df.to_dict("records"):
        close = parse_price(row["發行量加權股價指數"])
        if close is None:
            raise SchemaChangedError(f"TWSE FMTQIK has no TAIEX close on {row['date']}")
        records.append(
            {
                "index_id": IndexId.TAIEX.value,
                "date": row["date"],
                "close": close,
                "volume": parse_count(row["成交股數"]),
            }
        )
    return pd.DataFrame(records, columns=INDEX_COLUMNS, dtype=object)
