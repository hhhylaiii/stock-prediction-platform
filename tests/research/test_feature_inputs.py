import math
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from apps.market.models import DailyPrice, IndexDaily, SharesOutstanding, StockMaster
from apps.research.features.inputs import (
    FEATURE_WARMUP,
    attach_market_returns,
    attach_shares,
    load_feature_panel,
    load_index_returns,
    load_shares,
)
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2015, 1, 1), date(2016, 12, 31))
pytestmark = pytest.mark.django_db


def d(text: str) -> date:
    return date.fromisoformat(text)


def frame(stock_ids: list[str], days: list[str], market: str = "TWSE") -> pd.DataFrame:
    return pd.DataFrame({"stock_id": stock_ids, "date": [d(day) for day in days], "market": market})


def closes(index_id: str, values: dict[str, float]) -> None:
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id=index_id, date=d(day), close=Decimal(str(close)))
        for day, close in values.items()
    )


def test_shares_apply_from_their_effective_date() -> None:
    SharesOutstanding.objects.bulk_create(
        [
            SharesOutstanding(stock_id="2330", effective_date=d("2016-01-04"), shares=1_000),
            SharesOutstanding(stock_id="2330", effective_date=d("2016-03-01"), shares=2_000),
            SharesOutstanding(stock_id="2330", effective_date=d("2016-06-01"), shares=3_000),
        ]
    )
    panel = frame(
        ["2330", "2330", "2330", "1101"], ["2016-02-29", "2016-03-01", "2016-01-01", "2016-03-01"]
    )

    shares = load_shares(d("2016-03-31"))
    attached = attach_shares(panel, shares)

    assert set(shares["effective_date"]) == {d("2016-01-04"), d("2016-03-01")}
    assert attached["shares"].tolist()[:2] == [1_000.0, 2_000.0]
    assert attached["shares"].isna().tolist()[2:] == [True, True]
    assert attached[["stock_id", "date"]].equals(panel[["stock_id", "date"]])
    assert attach_shares(panel, shares.iloc[0:0])["shares"].isna().all()


def test_index_returns_need_closes_on_consecutive_trading_days() -> None:
    closes("TAIEX", {"2016-03-01": 100.0, "2016-03-02": 101.0, "2016-03-03": 99.99})
    closes("TPEX", {"2016-03-01": 50.0, "2016-03-03": 51.0, "2016-03-04": 51.51})

    returns = load_index_returns(CALENDAR, d("2016-03-02"), d("2016-03-04"))
    by_key = returns.set_index(["index_id", "date"])["market_ret"]

    assert by_key[("TAIEX", d("2016-03-02"))] == pytest.approx(0.01)
    assert by_key[("TAIEX", d("2016-03-03"))] == pytest.approx(-0.01)
    assert math.isnan(by_key[("TAIEX", d("2016-03-04"))])
    assert math.isnan(by_key[("TPEX", d("2016-03-03"))])
    assert by_key[("TPEX", d("2016-03-04"))] == pytest.approx(0.01)
    assert set(returns["index_id"]) == {"TAIEX", "TPEX"}


def test_each_market_takes_its_own_index() -> None:
    closes("TAIEX", {"2016-03-01": 100.0, "2016-03-02": 102.0})
    closes("TPEX", {"2016-03-01": 50.0, "2016-03-02": 49.0})
    panel = pd.concat(
        [
            frame(["2330"], ["2016-03-02"]),
            frame(["6488"], ["2016-03-02"], market="TPEx"),
            frame(["5450"], ["2016-03-02"], market=""),
        ],
        ignore_index=True,
    )

    attached = attach_market_returns(
        panel, load_index_returns(CALENDAR, d("2016-03-02"), d("2016-03-02"))
    )

    assert attached["market_ret"].tolist()[:2] == pytest.approx([0.02, -0.02])
    assert math.isnan(attached["market_ret"].iloc[2])


def test_feature_panel_keeps_warmup_rows_with_shares_and_market_returns() -> None:
    StockMaster.objects.create(stock_id="2330", name="2330", market="TWSE", industry="半導體業")
    days = CALENDAR.between(d("2015-01-02"), d("2016-12-30"))
    DailyPrice.objects.bulk_create(
        DailyPrice(
            stock_id="2330", date=day, open=10, high=10, low=10, close=10, volume=1, turnover=10
        )
        for day in days
    )
    SharesOutstanding.objects.create(stock_id="2330", effective_date=d("2015-01-02"), shares=500)
    closes("TAIEX", {day.isoformat(): 100.0 + k for k, day in enumerate(days)})

    panel = load_feature_panel(CALENDAR, d("2016-12-01"), d("2016-12-30"))

    assert panel["date"].iloc[0] == CALENDAR.shift(d("2016-12-01"), -FEATURE_WARMUP)
    assert panel["date"].iloc[-1] == d("2016-12-30")
    assert panel["shares"].eq(500.0).all()
    assert panel["market_ret"].iloc[-1] == pytest.approx(1 / (100.0 + len(days) - 2))
    assert {"in_universe", "limit_up_close", "attention", "turnover_med_20d"} <= set(panel.columns)


def test_inputs_at_the_edges_of_the_calendar_and_of_a_stock_filter() -> None:
    SharesOutstanding.objects.create(stock_id="2330", effective_date=d("2015-01-02"), shares=5)
    SharesOutstanding.objects.create(stock_id="1101", effective_date=d("2015-01-02"), shares=7)
    closes("TAIEX", {"2015-01-01": 100.0, "2015-01-02": 101.0})
    StockMaster.objects.create(stock_id="2330", name="2330", market="TWSE", industry="半導體業")
    DailyPrice.objects.create(
        stock_id="2330", date=d("2015-01-02"), open=1, high=1, low=1, close=1, volume=1, turnover=1
    )

    edge = load_index_returns(CALENDAR, d("2015-01-01"), d("2015-01-02"))
    panel = load_feature_panel(CALENDAR, d("2015-01-02"), d("2015-01-02"), ["2330"])

    assert load_shares(d("2015-12-31"), ["1101"])["shares"].tolist() == [7]
    assert load_index_returns(CALENDAR, d("2015-01-03"), d("2015-01-04")).empty
    taiex = edge[edge["index_id"] == "TAIEX"]["market_ret"].tolist()
    assert math.isnan(taiex[0])
    assert taiex[1] == pytest.approx(0.01)
    assert panel["date"].tolist() == [d("2015-01-02")]
    assert panel["shares"].tolist() == [5.0]
