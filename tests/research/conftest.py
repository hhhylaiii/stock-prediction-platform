from datetime import date, timedelta

import pytest

from apps.market.calendar import TradingDays


def weekday_calendar(start: date, end: date) -> TradingDays:
    """Synthetic calendar trading every Monday to Friday; long fold layouts need 2010 to 2026."""
    days = [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
        if (start + timedelta(days=offset)).weekday() < 5
    ]
    return TradingDays(days, start, end)


@pytest.fixture(scope="session")
def long_calendar() -> TradingDays:
    return weekday_calendar(date(2010, 1, 1), date(2026, 12, 31))
