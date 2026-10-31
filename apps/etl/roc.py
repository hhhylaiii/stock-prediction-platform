"""Republic of China calendar dates as published by TWSE and TPEx."""

import re
from datetime import date

from apps.etl.errors import SchemaChangedError

ROC_YEAR_OFFSET = 1911
ROC_DATE = re.compile(r"(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})")


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
