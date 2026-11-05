from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pytest
from django.core.cache import cache
from django.utils import timezone

from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.quality import CHECKS
from apps.market.models import Market, TradingCalendar
from tests.api.conftest import add_bars, add_stock

XSS_NAME = '<img src=x onerror="window.__xss=1">'
BAR_COUNT = 70
GAP_INDEX = 64


@dataclass(frozen=True)
class Seed:
    last_session: date
    closes: tuple[float | None, ...]


@pytest.fixture(scope="session", autouse=True)
def allow_orm_beside_playwright() -> Iterator[None]:
    """Playwright's sync API runs an event loop in the test thread, which Django would refuse."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DJANGO_ALLOW_ASYNC_UNSAFE", "true")
        yield


@pytest.fixture(autouse=True)
def fresh_throttle_cache() -> Iterator[None]:
    cache.clear()
    yield
    cache.clear()


def at(day: date, hour: int, minute: int) -> datetime:
    return timezone.make_aware(datetime.combine(day, time(hour, minute)))


@pytest.fixture
def seeded(transactional_db: None) -> Seed:
    """A calendar around today, two stocks, one day of ETL runs and a passing quality check."""
    today = timezone.localdate()
    days = [today + timedelta(days=n) for n in range(-200, 41)]
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5) for day in days
    )
    sessions = [day for day in days if day.weekday() < 5 and day <= today][-BAR_COUNT:]
    last = sessions[-1]
    closes = tuple(1000.0 + 2 * n for n in range(BAR_COUNT))
    add_stock()
    add_bars(
        "2330",
        [d for n, d in enumerate(sessions) if n != GAP_INDEX],
        [c for n, c in enumerate(closes) if n != GAP_INDEX],
    )
    add_stock(stock_id="9999", name=XSS_NAME, market=Market.TPEX)
    add_bars("9999", sessions[-3:], [10.0, 10.5, 11.0])
    for source, dataset, hour, minute, status in (
        ("twse", "daily_price", 15, 31, EtlRun.Status.SUCCESS),
        ("tpex", "daily_price", 15, 33, EtlRun.Status.SUCCESS),
        ("twse", "margin", 21, 31, EtlRun.Status.FAILED),
    ):
        EtlRun.objects.create(
            source=source,
            dataset=dataset,
            start_date=last,
            end_date=last,
            status=status,
            rows=1047,
            started_at=at(last, hour, minute),
        )
    EtlRun.objects.create(
        source="finmind",
        dataset="TaiwanStockPrice",
        start_date=date(2010, 1, 4),
        end_date=last,
        status=EtlRun.Status.SUCCESS,
        started_at=at(last, 9, 0),
    )
    DataQualityLog.objects.bulk_create(
        DataQualityLog(date=last, check_name=check, severity="info", detail={}) for check in CHECKS
    )
    return Seed(last_session=last, closes=closes)
