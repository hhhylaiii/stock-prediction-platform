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
    days = weekdays("2016-04-25", 30)
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
        (32 + 0.1 * memorial) / (32 + 0.1 * (memorial - 20)) - 1
    )


def test_usdtwd_waits_a_day_for_its_london_close() -> None:
    days = weekdays("2016-03-01", 25)
    rates = {day: 32.0 + 0.1 * k for k, day in enumerate(days)}
    closes = pd.concat([series("TSM_ADR", dict.fromkeys(days, 25.0)), series("USDTWD", rates)])
    tsmc = pd.Series(160.0, index=days)
    moved = pd.concat([closes, series("USDTWD", {days[22]: 99.0})]).drop_duplicates(
        ["index_id", "date"], keep="last"
    )

    features = market_features(days, closes, tsmc, no_signal())
    shifted = market_features(days, moved, tsmc, no_signal())

    assert value(features, days[22], "mkt_usdtwd_chg_20d") == pytest.approx(34.1 / 32.1 - 1)
    assert value(features, days[22], "mkt_tsm_adr_gap") == pytest.approx(25 * 34.1 / 5 / 160 - 1)
    pd.testing.assert_frame_equal(features.iloc[:23], shifted.iloc[:23])
    assert value(shifted, days[23], "mkt_tsm_adr_gap") == pytest.approx(25 * 99 / 5 / 160 - 1)


def test_adr_gap_converts_one_adr_to_five_shares() -> None:
    days = weekdays("2016-03-01", 3)
    closes = pd.concat(
        [
            series("TSM_ADR", dict.fromkeys(days, 25.0)),
            series("USDTWD", {d("2016-02-29"): 31.0, days[0]: 32.0, days[1]: 33.0}),
        ]
    )
    tsmc = pd.Series([150.0, NAN, 160.0], index=days)

    gap = market_features(days, closes, tsmc, no_signal())["mkt_tsm_adr_gap"].tolist()

    assert gap[0] == pytest.approx(25 * 31 / 5 / 150 - 1)
    assert math.isnan(gap[1])
    assert gap[2] == pytest.approx(25 * 33 / 5 / 160 - 1)


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
    days = weekdays("2016-03-01", 7)
    taiex = series("TAIEX", {day: 100.0 + k for k, day in enumerate(days)})
    market = market_features(days, taiex, pd.Series(dtype="float64"), no_signal())
    panel = pd.DataFrame(
        {"stock_id": ["2330", "1101", "2330"], "date": [days[5], days[5], days[6]], "close": 1.0}
    )

    attached = attach_market_features(panel, market)

    assert list(attached.columns) == ["stock_id", "date", "close", *feature_names(Group.MARKET)]
    assert attached["mkt_taiex_ret_5d"].tolist() == pytest.approx(
        [105 / 100 - 1, 105 / 100 - 1, 106 / 101 - 1]
    )


def test_us_closes_go_stale_after_seven_days() -> None:
    days = weekdays("2016-03-01", 9)
    closes = pd.concat(
        [
            series("TSM_ADR", {day: 25.0 for day in days if day <= d("2016-03-03")}),
            series("USDTWD", dict.fromkeys(days, 32.0)),
        ]
    )
    tsmc = pd.Series(160.0, index=days)

    gap = market_features(days, closes, tsmc, no_signal()).set_index("date")["mkt_tsm_adr_gap"]

    assert gap[d("2016-03-10")] == pytest.approx(25 * 32 / 5 / 160 - 1)
    assert math.isnan(gap[d("2016-03-11")])


def test_business_signal_goes_stale_after_62_days() -> None:
    days = [d("2016-03-29"), d("2016-03-30")]
    signal = pd.DataFrame(
        {"period": ["2015-12"], "value": [18.0], "announce_date": [d("2016-01-27")]}
    )

    values = market_features(days, series("TAIEX", {}), pd.Series(dtype="float64"), signal)

    assert values["mkt_business_signal"].tolist()[0] == 18.0
    assert math.isnan(values["mkt_business_signal"].iloc[1])


@pytest.mark.django_db
def test_loaded_series_are_warm_from_the_first_day() -> None:
    days = list(CALENDAR.between(d("2016-03-01"), d("2016-05-03")))
    steps = (("TAIEX", 8_000, "1"), ("USDTWD", 32, "0.1"), ("TSM_ADR", 25, "0"))
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id=index_id, date=day, close=base + Decimal(step) * k)
        for index_id, base, step in steps
        for k, day in enumerate(days)
    )
    StockMaster.objects.create(stock_id="2330", name="2330", market="TWSE", industry="半導體業")
    DailyPrice.objects.create(
        stock_id="2330",
        date=d("2016-05-02"),
        open=1,
        high=1,
        low=1,
        close=160,
        volume=1,
        turnover=1,
    )

    market = load_market_features(CALENDAR, d("2016-05-02"), d("2016-05-03"))

    first = days.index(d("2016-05-02"))
    rate = 32 + 0.1 * (first - 1)
    assert value(market, d("2016-05-02"), "mkt_taiex_ret_20d") == pytest.approx(
        (8_000 + first) / (8_000 + first - 20) - 1
    )
    assert value(market, d("2016-05-02"), "mkt_usdtwd_chg_20d") == pytest.approx(
        rate / (32 + 0.1 * (first - 21)) - 1
    )
    assert value(market, d("2016-05-02"), "mkt_tsm_adr_gap") == pytest.approx(
        25 * rate / 5 / 160 - 1
    )


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
