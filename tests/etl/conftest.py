import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import TradingCalendar
from tests.conftest import FIXTURES, load_trading_days

TWSE_LIMIT = RateLimit(
    min_interval=3.0, max_retries=2, backoff_base=1.0, backoff_max=2.0, cooldown=5.0
)


class RecordingSleep:
    """Stand-in for time.sleep that records requested delays instead of waiting."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def fake_sleep() -> RecordingSleep:
    return RecordingSleep()


@pytest.fixture
def twse_limiter(fake_sleep: RecordingSleep) -> RateLimiter:
    return RateLimiter(TWSE_LIMIT, sleep=fake_sleep)


@pytest.fixture(scope="session")
def october_2026() -> TradingDays:
    return load_trading_days("twse_2026_10.json")


@pytest.fixture
def october_2026_in_db(db: None, october_2026: TradingDays) -> TradingDays:
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=is_open)
        for day, is_open in october_2026.all_days()
    )
    return october_2026


@pytest.fixture
def load_fixture() -> Callable[[str], Any]:
    def _load(relative: str) -> Any:
        return json.loads((FIXTURES / relative).read_text(encoding="utf-8"))

    return _load


@pytest.fixture
def fixture_path() -> Callable[[str], Path]:
    return lambda relative: FIXTURES / relative
