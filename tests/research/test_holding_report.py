import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from apps.research.holding import Market
from apps.research.holding_report import (
    SUMMARY_COLUMNS,
    by_year,
    index_returns,
    summarise,
    universe_returns,
)
from tests.research.conftest import weekday_calendar

NAN = math.nan
DAYS = list(
    weekday_calendar(date(2016, 1, 4), date(2017, 3, 31)).between(
        date(2016, 1, 4), date(2017, 3, 31)
    )
)


def closes_market(closes: list[list[float]], universe: list[list[bool]]) -> Market:
    grid = np.array(closes, dtype=np.float64)
    days, names = grid.shape
    return Market(
        days=DAYS[:days],
        stocks=[str(1000 + k) for k in range(names)],
        opens=grid,
        closes=grid,
        score=np.ones((days, names)),
        in_universe=np.array(universe, dtype=bool),
        blocked=np.zeros((days, names), dtype=bool),
        delisted=np.full(names, days, dtype=np.int64),
        limit_down_open=np.zeros((days, names), dtype=bool),
    )


def growing(days: list[date], rate: float = 0.01) -> pd.DataFrame:
    value = (1 + rate) ** np.arange(len(days))
    return pd.DataFrame(
        {"date": days, "value": value, "cash": 0.0, "paid": 0.0, "traded": 0.0, "held": 30}
    )


def test_universe_returns_carry_closes_skip_names_never_closed_and_read_zero_when_empty() -> None:
    closes = [
        [10.0, NAN, 7.0],
        [11.0, NAN, 8.0],
        [NAN, 5.0, 9.0],
        [11.0, 5.5, 10.0],
        [12.0, 6.0, 11.0],
    ]
    universe = [[True, True, False]] * 3 + [[False, False, False]] + [[True, True, False]]

    out = universe_returns(closes_market(closes, universe))

    assert out == pytest.approx(np.array([0.0, 0.1, 0.0, 0.05, 0.0]))


def test_index_returns_carry_the_level_over_missing_closes() -> None:
    days = DAYS[:4]
    levels = pd.Series({days[0]: 100.0, days[1]: 110.0, days[3]: 121.0})

    assert index_returns(levels, days) == pytest.approx(np.array([0.0, 0.1, 0.0, 0.1]))


def test_summaries_scale_daily_means_to_months_and_compound_over_years() -> None:
    days = DAYS[:22]
    universe = np.full(len(DAYS), 0.002)
    index = np.full(len(DAYS), 0.001)
    daily = growing(days).assign(paid=0.0001, traded=0.2)

    row = summarise(daily, DAYS, universe, index)

    assert list(row) == SUMMARY_COLUMNS
    assert row["net_mo"] == pytest.approx(21.0)
    assert row["cost_mo"] == pytest.approx(np.mean(0.0001 / daily["value"][:-1]) * 21 * 100)
    assert row["ew_mo"] == pytest.approx(4.2)
    assert row["excess_mo"] == pytest.approx(16.8)
    assert math.isnan(row["excess_t"])
    assert row["index_mo"] == pytest.approx(2.1)
    assert row["net_cagr"] == pytest.approx((1.01**252 - 1) * 100)
    assert row["ew_cagr"] == pytest.approx((1.002**252 - 1) * 100)
    assert row["drawdown"] == pytest.approx(0.0)
    assert row["turnover_yr"] == pytest.approx(np.mean(0.1 / daily["value"][:-1]) * 252)
    assert row["held"] == 30
    assert row["days"] == 21


def test_each_book_return_meets_the_universe_and_index_return_of_the_same_day() -> None:
    first = 30
    days = DAYS[first : first + 5]
    universe = np.arange(len(DAYS), dtype=np.float64) * 1e-4
    index = np.arange(len(DAYS), dtype=np.float64) * 2e-4
    daily = growing(days, rate=0.0)

    row = summarise(daily, DAYS, universe, index)

    expected = universe[first + 1 : first + 5].mean() * 2100
    assert row["ew_mo"] == pytest.approx(expected)
    assert row["excess_mo"] == pytest.approx(-expected)
    assert row["index_mo"] == pytest.approx(index[first + 1 : first + 5].mean() * 2100)


def test_drawdown_is_the_worst_fall_from_a_running_peak() -> None:
    days = DAYS[:4]
    daily = growing(days).assign(value=[1.0, 1.2, 0.9, 1.1])

    row = summarise(daily, DAYS, np.zeros(len(DAYS)), np.zeros(len(DAYS)))

    assert row["drawdown"] == pytest.approx(-25.0)


def test_each_year_starts_from_the_last_value_of_the_year_before() -> None:
    first = DAYS.index(date(2016, 12, 29))
    days = DAYS[first : first + 6]
    daily = growing(days).assign(value=[1.0, 1.1, 1.21, 1.21, 1.331, 1.331])

    table = by_year(daily, DAYS, np.zeros(len(DAYS)), np.zeros(len(DAYS)))

    assert table.index.tolist() == ["2017"]
    assert table.loc["2017", "days"] == 4
    assert table.loc["2017", "net_mo"] == pytest.approx(np.mean([0.1, 0.0, 0.1, 0.0]) * 2100)
    assert table.columns.tolist() == SUMMARY_COLUMNS
