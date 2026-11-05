from collections.abc import Iterator, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pytest
from django.core.cache import cache

from apps.market.models import DailyPrice, Market, StockMaster, TradingCalendar

TAIPEI = ZoneInfo("Asia/Taipei")
CALENDAR_START = date(2026, 7, 1)
CALENDAR_END = date(2026, 11, 30)
HOLIDAYS = frozenset({date(2026, 9, 25), date(2026, 10, 9)})


def is_session(day: date) -> bool:
    return day.weekday() < 5 and day not in HOLIDAYS


def calendar_days(start: date = CALENDAR_START, end: date = CALENDAR_END) -> list[date]:
    return [start + timedelta(days=n) for n in range((end - start).days + 1)]


SESSIONS = [day for day in calendar_days() if is_session(day)]


def sessions_until(last: date) -> list[date]:
    return [day for day in SESSIONS if day <= last]


def taipei(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TAIPEI)


class JsonBody(Protocol):
    def json(self) -> Any: ...


def envelope_of(response: JsonBody) -> dict[str, Any]:
    """Decoded body after checking the four envelope keys are present."""
    body: dict[str, Any] = response.json()
    assert set(body) == {"success", "data", "error", "meta"}
    return body


@pytest.fixture(autouse=True)
def fresh_throttle_cache() -> Iterator[None]:
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def calendar_in_db(db: None) -> list[date]:
    """Weekdays of July to November 2026 minus two holidays, stored in trading_calendar."""
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=is_session(day)) for day in calendar_days()
    )
    return SESSIONS


def add_stock(
    stock_id: str = "2330",
    name: str = "台積電",
    market: str = Market.TWSE,
    industry: str = "半導體業",
    list_date: date | None = date(1994, 9, 5),
    delist_date: date | None = None,
) -> StockMaster:
    return StockMaster.objects.create(
        stock_id=stock_id,
        name=name,
        market=market,
        industry=industry,
        list_date=list_date,
        delist_date=delist_date,
    )


def add_bars(
    stock_id: str,
    days: Sequence[date],
    closes: Sequence[float | None],
    factors: Sequence[float] | None = None,
) -> list[DailyPrice]:
    """One bar per day around its close; a None close is a day without trades."""
    rows = []
    for index, (day, close) in enumerate(zip(days, closes, strict=True)):
        price = None if close is None else Decimal(str(close))
        rows.append(
            DailyPrice(
                stock_id=stock_id,
                date=day,
                open=None if price is None else price - 1,
                high=None if price is None else price + 2,
                low=None if price is None else price - 2,
                close=price,
                volume=0 if price is None else 1_000_000 + index,
                turnover=0 if price is None else int(price * 1_000_000),
                adj_factor=1.0 if factors is None else factors[index],
            )
        )
    return DailyPrice.objects.bulk_create(rows)
