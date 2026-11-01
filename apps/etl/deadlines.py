"""Conservative visibility dates for fundamentals until MOPS gives the real ones (SPEC §8.1)."""

import re
from datetime import date

from apps.market.calendar import TradingDays

MONTH = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
QUARTER = re.compile(r"^(\d{4})-Q([1-4])$")
REVENUE_DAY = 10
REVENUE_DAY_EXTENDED = 15
REVENUE_EXTENDED_FROM = date(2026, 1, 1)
INTERIM_DEADLINES = {1: (5, 30), 2: (8, 31), 3: (11, 29)}
ANNUAL_DEADLINE = (3, 31)
ANNUAL_DEADLINE_BEFORE_2011 = (4, 30)
THREE_MONTH_ANNUAL_FROM = 2011


def revenue_deadline(period: str) -> date:
    """Due the 10th of the next month; from 2026 insurers may file by the 15th."""
    found = MONTH.match(period)
    if found is None:
        raise ValueError(f"bad month period {period!r}")
    year, month = int(found[1]), int(found[2])
    due_month = date(year + month // 12, month % 12 + 1, 1)
    day = REVENUE_DAY_EXTENDED if due_month >= REVENUE_EXTENDED_FROM else REVENUE_DAY
    return due_month.replace(day=day)


def statement_deadline(period: str) -> date:
    """Latest legal deadline of any listed issuer; annual reports took 4 months before FY2011."""
    found = QUARTER.match(period)
    if found is None:
        raise ValueError(f"bad quarter period {period!r}")
    year, quarter = int(found[1]), int(found[2])
    if quarter in INTERIM_DEADLINES:
        return date(year, *INTERIM_DEADLINES[quarter])
    annual = ANNUAL_DEADLINE if year >= THREE_MONTH_ANNUAL_FROM else ANNUAL_DEADLINE_BEFORE_2011
    return date(year + 1, *annual)


def visible_from(deadline: date, calendar: TradingDays) -> date:
    """Trading day after the deadline, which itself moves past non-trading days."""
    base = calendar.latest_on_or_before(deadline)
    effective = base if base == deadline else calendar.shift(base, 1)
    return calendar.shift(effective, 1)
