"""Republic of China calendar dates as published by TWSE and TPEx."""

import re
from datetime import date
from typing import Any

from apps.etl.errors import SchemaChangedError

ROC_YEAR_OFFSET = 1911
ROC_DATE = re.compile(r"(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})")
FIRST_DAY_MARK = "*"


def parse_roc_date(text: str) -> date:
    """Parse '115/09/01', '090/01/20' or '81-10-27' into a Gregorian date."""
    match = ROC_DATE.fullmatch(str(text).strip())
    if match is None:
        raise SchemaChangedError(f"not a ROC date: {text!r}")
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year + ROC_YEAR_OFFSET, month, day)
    except ValueError:
        raise SchemaChangedError(f"not a ROC date: {text!r}") from None


def days_in_month(rows: Any, month: date, label: str) -> list[date]:
    """First-column dates of a per-stock monthly table; TPEx marks a first listed day with '*'."""
    if not isinstance(rows, list) or not all(isinstance(row, list) and row for row in rows):
        raise SchemaChangedError(f"{label} rows are malformed")
    days = sorted(parse_roc_date(str(row[0]).rstrip(FIRST_DAY_MARK)) for row in rows)
    if any(day.replace(day=1) != month for day in days):
        raise SchemaChangedError(f"{label} answered {month:%Y-%m} with days of another month")
    return days
