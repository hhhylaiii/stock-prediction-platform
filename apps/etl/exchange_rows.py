"""Values and per-stock monthly tables as TWSE and TPEx publish them."""

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, NamedTuple

from apps.etl.errors import SchemaChangedError
from apps.etl.roc import parse_roc_date

NO_TRADE_MARKERS = frozenset({"--", "---", ""})
FIRST_DAY_MARK = "*"
VOLUME_AT = 1
CLOSE_AT = 6


class OfficialDay(NamedTuple):
    day: date
    close: Decimal | None
    volume: int
    volume_unit: int


def parse_price(text: str) -> Decimal | None:
    cleaned = str(text).replace(",", "").strip()
    if cleaned in NO_TRADE_MARKERS:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        raise SchemaChangedError(f"unexpected price value {text!r}") from None


def parse_count(text: str) -> int:
    cleaned = str(text).replace(",", "").strip()
    if not cleaned.isdigit():
        raise SchemaChangedError(f"unexpected count value {text!r}")
    return int(cleaned)


def official_days(rows: Any, month: date, label: str, volume_unit: int) -> list[OfficialDay]:
    """One month of a per-stock table; TPEx counts thousands of shares and marks a first day '*'."""
    if not isinstance(rows, list) or not all(
        isinstance(row, list) and len(row) > CLOSE_AT for row in rows
    ):
        raise SchemaChangedError(f"{label} rows are malformed")
    days = sorted(
        OfficialDay(
            parse_roc_date(str(row[0]).rstrip(FIRST_DAY_MARK)),
            parse_price(row[CLOSE_AT]),
            parse_count(row[VOLUME_AT]) * volume_unit,
            volume_unit,
        )
        for row in rows
    )
    if any(official.day.replace(day=1) != month for official in days):
        raise SchemaChangedError(f"{label} answered {month:%Y-%m} with days of another month")
    return days
