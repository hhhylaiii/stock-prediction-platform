import math
from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from apps.research.pricelimit import (
    apply_price_limits,
    limit_down_price,
    limit_up_price,
    tick_size,
)

BEFORE, AFTER = date(2015, 5, 29), date(2015, 6, 1)


def d(text: str) -> date:
    return date.fromisoformat(text)


@pytest.mark.parametrize(
    ("price", "tick"),
    [
        (9.99, 0.01),
        (10.0, 0.05),
        (49.95, 0.05),
        (50.0, 0.1),
        (99.9, 0.1),
        (100.0, 0.5),
        (499.5, 0.5),
        (500.0, 1.0),
        (999.0, 1.0),
        (1000.0, 5.0),
    ],
)
def test_tick_size_follows_the_price_band(price: float, tick: float) -> None:
    assert tick_size(np.array([price]))[0] == tick


@pytest.mark.parametrize(
    ("reference", "day", "expected"),
    [
        (100.0, AFTER, 90.0),
        (100.0, BEFORE, 93.0),
        (11.3, AFTER, 10.2),
        (10.6, AFTER, 9.54),
        (55.5, AFTER, 49.95),
        (1150.0, AFTER, 1035.0),
    ],
)
def test_limit_down_rounds_up_to_the_tick(reference: float, day: date, expected: float) -> None:
    assert limit_down_price(np.array([reference]), [day])[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("reference", "expected"),
    [(45.5, 50.0), (9.2, 10.1), (998.0, 1095.0), (9.09, 9.99)],
)
def test_limit_up_uses_the_tick_of_the_limit_price_band(reference: float, expected: float) -> None:
    assert limit_up_price(np.array([reference]), [AFTER])[0] == pytest.approx(expected)


def bars(**columns: Sequence[object]) -> pd.DataFrame:
    """One stock, one listing, long listed, every price 100 with factor 1 unless overridden."""
    n = len(next(iter(columns.values()))) if columns else 3
    base: dict[str, object] = {
        "stock_id": "2330",
        "segment": 1,
        "date": [d("2016-03-01") + timedelta(days=i) for i in range(n)],
        "age": np.arange(n) + 100,
        "open": 100.0,
        "close": 100.0,
        "adj_factor": 1.0,
    }
    frame = pd.DataFrame({**base, **columns})
    frame["adj_close"] = frame["close"] * frame["adj_factor"]
    return frame


def test_close_and_open_at_the_limit_are_flagged() -> None:
    result = apply_price_limits(bars(close=[100.0, 110.0, 99.0], open=[100.0, 105.0, 121.0]))
    assert result["limit_up"].tolist()[1:] == [110.0, 121.0]
    assert result["limit_down"].tolist()[1:] == [90.0, 99.0]
    assert result["limit_up_close"].tolist() == [False, True, False]
    assert result["limit_down_close"].tolist() == [False, False, True]
    assert result["limit_up_open"].tolist() == [False, False, True]


def test_ex_date_reference_is_the_adjusted_previous_close() -> None:
    result = apply_price_limits(bars(close=[100.0, 104.5], adj_factor=[0.95, 1.0]))
    assert result.loc[1, "reference"] == pytest.approx(95.0)
    assert result.loc[1, "limit_up"] == pytest.approx(104.5)
    assert result.loc[1, "limit_up_close"]


def test_reference_carries_the_last_close_over_days_without_trades() -> None:
    result = apply_price_limits(bars(close=[50.0, math.nan, math.nan, 55.0]))
    assert result["reference"].tolist()[1:] == [50.0, 50.0, 50.0]
    assert result.loc[3, "limit_up_close"]


def test_first_five_days_of_a_listing_have_no_limits() -> None:
    close = [10.0, 15.0, 22.0, 30.0, 40.0, 44.0, 48.4]
    result = apply_price_limits(bars(close=close, age=list(range(7))))
    assert result.loc[:4, "limit_up"].isna().all()
    assert not result.loc[:4, "limit_up_close"].any()
    assert result.loc[5:, "limit_up_close"].tolist() == [True, True]


def test_a_new_listing_does_not_inherit_the_old_close() -> None:
    frame = bars(close=[100.0, 100.0, 30.0], segment=[1, 1, 2], age=[200, 201, 300])
    result = apply_price_limits(frame)
    assert result["reference"].isna().tolist() == [True, False, True]
    assert not result.loc[2, "limit_down_close"]


def test_stocks_do_not_share_references() -> None:
    first, second = bars(close=[100.0, 110.0]), bars(close=[10.0, 11.0])
    second["stock_id"] = "2317"
    result = apply_price_limits(pd.concat([first, second], ignore_index=True))
    assert result["reference"].tolist()[2:] == [pytest.approx(float("nan"), nan_ok=True), 10.0]
    assert result["limit_up_close"].tolist() == [False, True, False, True]


def test_input_is_not_modified() -> None:
    frame = bars()
    before = frame.copy()
    apply_price_limits(frame)
    pd.testing.assert_frame_equal(frame, before)
