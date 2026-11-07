"""Republic of China calendar dates as published by TWSE and TPEx."""

import re
from datetime import date

from apps.etl.errors import SchemaChangedError

ROC_YEAR_OFFSET = 1911
ROC_DATE = re.compile(r"(\d{2,3})([/.-])(\d{1,2})\2(\d{1,2})")


def parse_roc_date(text: str) -> date:
    """Parse '115/09/01', '090/01/20', '81-10-27' or '99.01.04' into a Gregorian date."""
    match = ROC_DATE.fullmatch(str(text).strip())
    if match is None:
        raise SchemaChangedError(f"not a ROC date: {text!r}")
    year, month, day = (int(match.group(index)) for index in (1, 3, 4))
    try:
        return date(year + ROC_YEAR_OFFSET, month, day)
    except ValueError:
        raise SchemaChangedError(f"not a ROC date: {text!r}") from None


ROC_WRITTEN = re.compile(r"(\d{2,3})年(\d{2})月(\d{2})日")
ROC_COMPACT = re.compile(r"(\d{2,3})(\d{2})(\d{2})")


def parse_roc_day(text: str) -> date:
    """Parse '115/03/17', '115年03月17日' or the separator-free '1150317'."""
    cleaned = str(text).strip()
    match = ROC_WRITTEN.fullmatch(cleaned) or ROC_COMPACT.fullmatch(cleaned)
    if match is None:
        return parse_roc_date(cleaned)
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year + ROC_YEAR_OFFSET, month, day)
    except ValueError:
        raise SchemaChangedError(f"not a ROC date: {text!r}") from None
