import math
from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from apps.market.models import DailyPrice, IndexDaily, MacroIndicator, StockMaster
from apps.research.features.market import (
    BUSINESS_SIGNAL_ID,
    attach_market_features,
    load_market_features,
    market_features,
)
from apps.research.features.registry import Group, feature_names
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2016, 1, 1), date(2016, 12, 31))
NAN = math.nan


def d(text: str) -> date:
    return date.fromisoformat(text)


def series(index_id: str, closes: dict[date, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"index_id": index_id, "date": list(closes), "close": list(closes.values())}
    )


def weekdays(start: str, n: int) -> list[date]:
    return list(CALENDAR.between(d(start), CALENDAR.end))[:n]


def value(frame: pd.DataFrame, day: date, column: str) -> float:
    return float(frame.loc[frame["date"] == day, column].iloc[0])


def no_signal() -> pd.DataFrame:
    return pd.DataFrame(columns=["period", "value", "announce_date"])


def test_taiex_returns_count_trading_days() -> None:
    days = weekdays("2016-03-01", 30)
    taiex = series("TAIEX", {day: 100.0 + k for k, day in enumerate(days)})

    features = market_features(days, taiex, pd.Series(dtype="float64"), no_signal())

    assert list(features.columns) == ["date", *feature_names(Group.MARKET)]
    assert features["mkt_taiex_ret_5d"].iloc[25] == pytest.approx(125 / 120 - 1)
    assert features["mkt_taiex_ret_20d"].iloc[25] == pytest.approx(125 / 105 - 1)
    assert math.isnan(features["mkt_taiex_ret_20d"].iloc[19])


def test_us_series_use_their_own_trading_days_up_to_the_taipei_date() -> None:
    days = weekdays("2016-05-02", 25)
    us_days = [day for day in days if day != d("2016-05-30")]
    closes = pd.concat(
        [
            series("SOX", {day: 700.0 + 10 * k for k, day in enumerate(us_days)}),
            series("USDTWD", {day: 32.0 + 0.1 * k for k, day in enumerate(us_days)}),
        ]
    )

    features = market_features(days, closes, pd.Series(dtype="float64"), no_signal())

    memorial = us_days.index(d("2016-05-27"))
    assert value(features, d("2016-05-30"), "mkt_sox_ret_5d") == pytest.approx(
        (700 + 10 * memorial) / (700 + 10 * (memorial - 5)) - 1
    )
    after = us_days.index(d("2016-05-31"))
    assert value(features, d("2016-05-31"), "mkt_sox_ret_5d") == pytest.approx(
        (700 + 10 * after) / (700 + 10 * (after - 5)) - 1
    )
    assert value(features, d("2016-05-31"), "mkt_usdtwd_chg_20d") == pytest.approx(
        (32 + 0.1 * after) / (32 + 0.1 * (after - 20)) - 1
    )


def test_adr_gap_converts_one_adr_to_five_shares() -> None:
    days = weekdays("2016-03-01", 3)
    closes = pd.concat(
        [
            series("TSM_ADR", dict.fromkeys(days, 25.0)),
            series("USDTWD", dict.fromkeys(days, 32.0)),
        ]
    )
    tsmc = pd.Series([150.0, NAN, 160.0], index=days)

    gap = market_features(days, closes, tsmc, no_signal())["mkt_tsm_adr_gap"].tolist()

    assert gap[0] == pytest.approx(25 * 32 / 5 / 150 - 1)
    assert math.isnan(gap[1])
    assert gap[2] == pytest.approx(25 * 32 / 5 / 160 - 1)


def test_business_signal_waits_for_its_announcement() -> None:
    days = weekdays("2016-02-25", 5)
    signal = pd.DataFrame(
        {
            "period": ["2015-12", "2016-01"],
            "value": [18.0, 21.0],
            "announce_date": [d("2016-01-27"), d("2016-02-29")],
        }
    )

    values = market_features(days, series("TAIEX", {}), pd.Series(dtype="float64"), signal)

    assert values.set_index("date")["mkt_business_signal"].tolist() == [
        18.0,
        18.0,
        21.0,
        21.0,
        21.0,
    ]


def test_market_values_are_shared_by_every_stock_of_the_day() -> None:
    days = weekdays("2016-03-01", 2)
    market = market_features(days, series("TAIEX", {}), pd.Series(dtype="float64"), no_signal())
    panel = pd.DataFrame(
        {"stock_id": ["2330", "1101", "2330"], "date": [days[0], days[0], days[1]]}
    )

    features = attach_market_features(panel, market)

    assert list(features.columns) == ["stock_id", "date", *feature_names(Group.MARKET)]
    assert features["stock_id"].tolist() == ["2330", "1101", "2330"]


@pytest.mark.django_db
@pytest.mark.spec8
def test_sox_return_uses_the_us_close_of_the_same_date_and_the_day_before_on_holidays() -> None:
    days = list(CALENDAR.between(d("2016-04-01"), d("2016-06-01")))
    us_days = [day for day in days if day != d("2016-05-30")]
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id="SOX", date=day, close=Decimal(700 + 10 * k))
        for k, day in enumerate(us_days)
    )
    StockMaster.objects.create(stock_id="2330", name="2330", market="TWSE", industry="半導體業")
    DailyPrice.objects.create(
        stock_id="2330", date=d("2016-05-31"), open=1, high=1, low=1, close=1, volume=1, turnover=1
    )
    MacroIndicator.objects.create(
        indicator_id=BUSINESS_SIGNAL_ID, period="2016-04", value=20.0, announce_date=d("2016-05-27")
    )

    market = load_market_features(CALENDAR, d("2016-05-30"), d("2016-05-31"))

    friday, tuesday = us_days.index(d("2016-05-27")), us_days.index(d("2016-05-31"))
    assert market["date"].tolist() == [d("2016-05-30"), d("2016-05-31")]
    assert value(market, d("2016-05-30"), "mkt_sox_ret_5d") == pytest.approx(
        (700 + 10 * friday) / (700 + 10 * (friday - 5)) - 1
    )
    assert value(market, d("2016-05-31"), "mkt_sox_ret_5d") == pytest.approx(
        (700 + 10 * tuesday) / (700 + 10 * (tuesday - 5)) - 1
    )
    assert market["mkt_business_signal"].tolist() == [20.0, 20.0]
    assert d("2016-05-31") + timedelta(days=1) in us_days
