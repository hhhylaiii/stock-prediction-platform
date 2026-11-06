import math
from datetime import date, timedelta

import pandas as pd
import pytest

from apps.market.calendar import TradingDays
from apps.market.models import IndexDaily
from apps.research.relative_up import market_digest, market_returns, relative_up

pytestmark = pytest.mark.django_db

START = date(2016, 1, 4)


def sessions(count: int) -> list[date]:
    days = (START + timedelta(days=n) for n in range(count * 2))
    return [day for day in days if day.weekday() < 5][:count]


def calendar(days: list[date]) -> TradingDays:
    return TradingDays(days, days[0], days[-1])


def store(days: list[date], closes: list[float], index_id: str = "TAIEX") -> None:
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id=index_id, date=day, close=close)
        for day, close in zip(days, closes, strict=True)
    )


def test_market_returns_run_from_the_close_of_t_to_the_close_twenty_sessions_later() -> None:
    days = sessions(30)
    closes = [round(100 * 1.01**k, 4) for k in range(30)]
    store(days, closes)
    store(days, [50.0] * 30, "TPEX")

    returns = market_returns(calendar(days), days[:12])

    assert returns[days[0]] == pytest.approx(closes[20] / closes[0] - 1)
    assert returns[days[9]] == pytest.approx(closes[29] / closes[9] - 1)
    assert math.isnan(returns[days[10]])
    assert list(returns.index) == days[:12]


def test_market_returns_of_no_days_are_empty() -> None:
    days = sessions(30)

    returns = market_returns(calendar(days), [])

    assert returns.empty


def test_market_returns_need_both_closes() -> None:
    days = sessions(30)
    store([day for k, day in enumerate(days) if k not in (1, 22)], [100.0] * 28)

    returns = market_returns(calendar(days), days[:3])

    assert returns[days[0]] == pytest.approx(0.0)
    assert math.isnan(returns[days[1]])
    assert math.isnan(returns[days[2]])


def test_relative_up_flags_twenty_day_returns_beyond_the_market_by_five_percent() -> None:
    days = sessions(30)
    store(days, [100.0] * 20 + [110.0] * 10)
    labels = pd.DataFrame(
        {
            "stock_id": ["1101", "1102", "1103", "1104", "1105"],
            "date": [days[0], days[0], days[0], days[0], days[10]],
            "fwd_ret_open": [0.151, 0.149, math.nan, 0.30, 0.30],
        },
        index=[7, 8, 9, 10, 11],
    )

    flags = relative_up(labels, calendar(days))

    assert flags.to_dict() == {7: True, 8: False, 9: None, 10: True, 11: None}


def test_the_market_digest_follows_the_taiex_closes_of_its_range() -> None:
    days = sessions(30)
    store(days, [100.0 + k for k in range(30)])
    first = market_digest(days[0], days[20])

    IndexDaily.objects.filter(index_id="TAIEX", date=days[25]).update(close=1)
    store(days, [1.0] * 30, "TPEX")
    assert market_digest(days[0], days[20]) == first

    IndexDaily.objects.filter(index_id="TAIEX", date=days[20]).update(close=1)
    assert market_digest(days[0], days[20]) != first
