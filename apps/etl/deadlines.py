"""Conservative visibility dates for fundamentals until MOPS gives the real ones (SPEC §8.1)."""

import re
from datetime import date

from apps.market.calendar import TradingDays

MONTH = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
QUARTER = re.compile(r"^(\d{4})-Q([1-4])$")
REVENUE_DAY = 10
REVENUE_DAY_EXTENDED = 15
REVENUE_EXTENDED_FROM = date(2026, 1, 1)
INTERIM_DEADLINES = {1: (5, 31), 2: (8, 31), 3: (11, 30)}
WEEKEND = 5
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
    """End of the second month for Q1–Q3 (latest issuer); annual took 4 months before FY2011."""
    found = QUARTER.match(period)
    if found is None:
        raise ValueError(f"bad quarter period {period!r}")
    year, quarter = int(found[1]), int(found[2])
    if quarter in INTERIM_DEADLINES:
        return date(year, *INTERIM_DEADLINES[quarter])
    annual = ANNUAL_DEADLINE if year >= THREE_MONTH_ANNUAL_FROM else ANNUAL_DEADLINE_BEFORE_2011
    return date(year + 1, *annual)


def visible_from(deadline: date, calendar: TradingDays, extension: int = 0) -> date:
    """Trading day after the deadline, which moves past non-trading days and extension sessions."""
    base = calendar.latest_on_or_before(deadline)
    effective = base if base == deadline else calendar.shift(base, 1)
    return calendar.shift(effective, extension + 1)


def revenue_visible_from(period: str, calendar: TradingDays) -> date:
    """FSC extends the deadline after early-month holidays; one session per closed weekday."""
    deadline = revenue_deadline(period)
    window = (deadline.replace(day=day) for day in range(1, deadline.day))
    closed = sum(
        1 for day in window if day.weekday() < WEEKEND and not calendar.is_trading_day(day)
    )
    return visible_from(deadline, calendar, closed)
