import json
from datetime import date
from pathlib import Path

import pytest

from apps.market.calendar import TradingDays
from tests import spec8_guard

FIXTURES = Path(__file__).parent / "fixtures"

pytest_runtest_makereport = spec8_guard.pytest_runtest_makereport


def load_trading_days(name: str) -> TradingDays:
    """Build TradingDays from a recorded calendar fixture under tests/fixtures/calendar."""
    data = json.loads((FIXTURES / "calendar" / name).read_text(encoding="utf-8"))
    return TradingDays(
        (date.fromisoformat(d) for d in data["trading_days"]),
        date.fromisoformat(data["start"]),
        date.fromisoformat(data["end"]),
    )


@pytest.fixture(scope="session")
def twse_calendar() -> TradingDays:
    return load_trading_days("twse_2014_2016.json")
