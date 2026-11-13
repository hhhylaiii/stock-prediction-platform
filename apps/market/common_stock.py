"""Which listed codes count as common stocks, shared by the research panel and the data checks."""

import re

COMMON_STOCK_PATTERN = "^[1-9][0-9]{3}$"
DEPOSITARY_PREFIX = "91%"
DEPOSITARY_INDUSTRY = "存託憑證"


def is_common_stock(stock_id: str, industry: str) -> bool:
    """The filters above for one code: four digits, not a depositary receipt."""
    return (
        re.fullmatch(COMMON_STOCK_PATTERN, stock_id) is not None
        and not stock_id.startswith(DEPOSITARY_PREFIX.removesuffix("%"))
        and industry != DEPOSITARY_INDUSTRY
    )
