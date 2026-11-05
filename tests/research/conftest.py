from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from apps.market.calendar import TradingDays
from tests.research.run_panels import build_panel


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


@pytest.fixture
def run_panel(settings: Any, tmp_path: Path, db: None) -> Path:
    """Feature partitions, labels and a screening for the walk-forward runner tests."""
    return build_panel(settings, tmp_path)
