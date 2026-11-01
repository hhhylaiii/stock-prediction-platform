import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from apps.research.labels import label_panel
from apps.research.metrics.deciles import (
    BUY_COST,
    SELL_COST,
    SUMMARY_COLUMNS,
    UNIVERSE,
    decile_groups,
    top_liquidity,
    tranche_returns,
    tranche_summary,
)
from tests.research.conftest import weekday_calendar

NAN = math.nan
CALENDAR = weekday_calendar(date(2016, 1, 1), date(2016, 3, 31))
DAYS = CALENDAR.between(date(2016, 1, 4), date(2016, 3, 31))
Bars = dict[str, list[tuple[float, float]]]


def prices(table: Bars) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stock_id": stock, "date": DAYS[k], "adj_open": bar[0], "adj_close": bar[1]}
            for stock, bars in table.items()
            for k, bar in enumerate(bars)
        ]
    )


def signal(*days: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": DAYS[k], "stock_id": stock, "score": score}
            for k, scores in enumerate(days)
            for stock, score in scores.items()
        ]
    )


def series(daily: pd.DataFrame, portfolio: str, column: str) -> list[float]:
    return daily.loc[daily["portfolio"] == portfolio, column].tolist()


def test_groups_split_each_day_into_equal_counts_with_the_top_scores_last() -> None:
    first = {f"{1000 + k}": float(k) for k in range(20)}
    second = {f"{2000 + k}": float(k // 2) for k in range(10)}
    frame = signal(first, second).sample(frac=1.0, random_state=7)

    groups = decile_groups(frame)

    expected = [
        int(score) // 2 + 1 if day == DAYS[0] else int(stock) - 2000 + 1
        for day, stock, score in frame[["date", "stock_id", "score"]].itertuples(index=False)
    ]
    assert groups.index.equals(frame.index)
    assert groups.tolist() == expected


def test_a_tranche_buys_at_the_next_open_and_holds_to_the_horizon_close() -> None:
    bars = prices(
        {
            "1101": [(9, 9.5), (10, 11), (11.5, 11), (12, 12)],
            "2330": [(19, 19), (20, 20), (21, 30), (30, 31)],
        }
    )

    daily = tranche_returns(signal({"1101": 1.0, "2330": 2.0}), bars, CALENDAR, 2, groups=1)

    assert sorted(set(daily["date"])) == [DAYS[1], DAYS[2]]
    gross = series(daily, "D1", "gross")
    assert gross == pytest.approx([0.05 / 2, (1.3 / 1.05 - 1) / 2])
    assert series(daily, "D1", "sleeves") == [1, 1]
    assert series(daily, UNIVERSE, "gross") == pytest.approx(gross)
    assert (1 + 2 * gross[0]) * (1 + 2 * gross[1]) - 1 == pytest.approx(0.3)


def test_the_tranche_return_is_the_mean_open_label_through_halts_and_delistings() -> None:
    closes: dict[str, list[float]] = {
        "1101": [10, 11, 12, NAN, 13, 12, 14, 15],
        "2330": [50, 52, 51, 53, 55, 54, 56, 57],
        "2002": [20, 21, 22, 23],
    }
    panel = pd.DataFrame(
        [
            {
                "stock_id": stock,
                "segment": 0,
                "date": DAYS[k],
                "adj_open": close - 0.5 if not math.isnan(close) else NAN,
                "adj_close": close,
                "limit_up_open": False,
                "in_universe": True,
            }
            for stock, values in closes.items()
            for k, close in enumerate(values)
        ]
    ).sort_values(["stock_id", "date"], ignore_index=True)
    labels = label_panel(panel, horizons=(5,))
    opening = labels[labels["date"] == DAYS[0]]

    daily = tranche_returns(
        opening.assign(score=1.0)[["date", "stock_id", "score"]],
        panel[["stock_id", "date", "adj_open", "adj_close"]],
        CALENDAR,
        5,
        groups=1,
    )

    growth = np.prod([1 + 5 * value for value in series(daily, "D1", "gross")]) - 1
    assert len(daily[daily["portfolio"] == "D1"]) == 5
    assert growth == pytest.approx(opening["fwd_ret_open"].mean())


def test_overlapping_sleeves_average_the_live_tranches() -> None:
    o = [10.0, 11.0, 12.0, 13.0, 14.0]
    c = [10.5, 11.5, 11.0, 13.5, 13.0]
    bars = prices({"1101": list(zip(o, c, strict=True))})

    daily = tranche_returns(
        signal({"1101": 1.0}, {"1101": 1.0}, {"1101": 1.0}), bars, CALENDAR, 2, groups=1
    )

    assert series(daily, "D1", "gross") == pytest.approx(
        [
            (c[1] / o[1] - 1) / 2,
            ((c[2] / c[1] - 1) + (c[2] / o[2] - 1)) / 2,
            ((c[3] / c[2] - 1) + (c[3] / o[3] - 1)) / 2,
            (c[4] / c[3] - 1) / 2,
        ]
    )
    assert series(daily, "D1", "sleeves") == [1, 2, 2, 1]


def test_costs_apply_only_to_the_traded_weights() -> None:
    bars = prices(
        {
            "1101": [(10, 10), (10, 12), (12, 12)],
            "2330": [(10, 10), (10, 8), (8, 8.8)],
            "2002": [(5, 5), (5, 5), (5, 5)],
        }
    )

    daily = tranche_returns(
        signal({"1101": 1.0, "2330": 1.0}, {"2330": 1.0, "2002": 1.0}), bars, CALENDAR, 1, groups=1
    )

    assert series(daily, "D1", "gross") == pytest.approx([0.0, 0.05])
    assert series(daily, "D1", "turnover") == pytest.approx([0.8, 0.8])
    cost = [BUY_COST + 0.6 * SELL_COST, 0.6 * BUY_COST + SELL_COST]
    assert series(daily, "D1", "cost") == pytest.approx(cost)
    rates = (BUY_COST, SELL_COST)
    assert rates == pytest.approx((0.001425, 0.001425 + 0.003))
    assert series(daily, "D1", "net") == pytest.approx([0.0 - cost[0], 0.05 - cost[1]])


def test_costs_and_turnover_spread_over_the_sleeves_of_a_longer_horizon() -> None:
    bars = prices(
        {
            "1101": [(10, 10), (10, 11), (11, 12), (12, 12), (12, 13)],
            "2330": [(10, 10), (10, 9), (9, 8), (8, 8), (8, 8)],
            "2002": [(5, 5)] * 5,
        }
    )
    pair = {"1101": 1.0, "2330": 2.0}

    daily = tranche_returns(
        signal(pair, {"2330": 1.0, "2002": 2.0}, pair), bars, CALENDAR, 2, groups=1
    )

    assert series(daily, "D1", "sleeves") == [1, 2, 2, 1]
    assert series(daily, "D1", "turnover") == pytest.approx([0.25, 0.275, 0.275, 0.25])
    assert series(daily, "D1", "cost") == pytest.approx(
        [
            BUY_COST / 2,
            (BUY_COST + 0.1 * SELL_COST) / 2,
            (0.1 * BUY_COST + SELL_COST) / 2,
            SELL_COST / 2,
        ]
    )
    assert tranche_summary(daily, 2).empty
    with pytest.raises(ValueError, match="horizon"):
        tranche_summary(daily, 1)


def test_holding_the_same_names_costs_nothing_to_roll() -> None:
    bars = prices({"1101": [(10, 10)] * 3, "2330": [(10, 10)] * 3})
    both = {"1101": 1.0, "2330": 2.0}

    daily = tranche_returns(signal(both, both), bars, CALENDAR, 1, groups=1)

    assert series(daily, "D1", "cost") == pytest.approx([BUY_COST, SELL_COST])
    assert series(daily, "D1", "turnover") == pytest.approx([0.5, 0.5])


def test_the_universe_holds_every_signal_row() -> None:
    moves = {"1101": 1.1, "2330": 1.2, "2002": 0.9, "1301": 1.0, "2412": 1.5}
    bars = prices({stock: [(10, 10), (10, 10 * move)] for stock, move in moves.items()})
    scores = {"1101": 1.0, "2330": 2.0, "2002": 3.0, "1301": 4.0, "2412": NAN}

    daily = tranche_returns(signal(scores), bars, CALENDAR, 1, groups=2)

    assert set(daily["portfolio"]) == {"D1", "D2", UNIVERSE}
    assert series(daily, "D1", "gross") == pytest.approx([0.15])
    assert series(daily, "D2", "gross") == pytest.approx([-0.05])
    assert series(daily, UNIVERSE, "gross") == pytest.approx([0.05])


@pytest.mark.parametrize(
    ("frame", "horizon", "message"),
    [
        (signal({"1101": 1.0}).assign(date=date(2016, 1, 9)), 1, "not a trading day"),
        (signal({}, {"1101": 1.0}), 1, "no open"),
        (signal({}, {}, {}, {"1101": 1.0}), 1, "no open"),
        (signal({"1101": 1.0}, {}, {}, {}, {}, {"1101": 1.0}), 1, "no open"),
        (signal({"1101": 1.0}), 0, "horizon"),
        (pd.concat([signal({"1101": 1.0})] * 2, ignore_index=True), 1, "more than one row"),
        (signal({}, {}, {}, {}, {"1101": 1.0}), 1, "no open"),
    ],
)
def test_bad_signals_are_refused(frame: pd.DataFrame, horizon: int, message: str) -> None:
    bars = prices({"1101": [(10, 10), (10, 10), (NAN, NAN), (10, 10), (10, NAN)]})

    with pytest.raises(ValueError, match=message):
        tranche_returns(frame, bars, CALENDAR, horizon, groups=1)


def test_the_inputs_stay_unchanged() -> None:
    bars = prices({"1101": [(10, 10), (10, 11), (11, 12)], "2330": [(5, 5), (5, 6), (6, 6)]})
    frame = signal({"2330": 2.0, "1101": 1.0})
    before = (frame.copy(), bars.copy())

    daily = tranche_returns(frame, bars, CALENDAR, 2, groups=1)
    decile_groups(frame)
    before_daily = daily.copy()
    tranche_summary(daily, 2)
    liquid = frame.assign(liq_dollar_vol_med_20d=1.0)
    before_liquid = liquid.copy()
    top_liquidity(liquid)
    pd.testing.assert_frame_equal(daily, before_daily)
    pd.testing.assert_frame_equal(liquid, before_liquid)

    pd.testing.assert_frame_equal(frame, before[0])
    pd.testing.assert_frame_equal(bars, before[1])


def test_the_summary_skips_building_and_unwinding_days() -> None:
    days = CALENDAR.between(date(2016, 1, 26), date(2016, 2, 4))
    sleeves = [1, 2, 2, 2, 2, 2, 2, 1]
    edge = [0.5, 0.5]
    nets = {
        "D10": [*edge, -0.04, 0.01, 0.02, 0.0, *edge],
        UNIVERSE: [*edge, 0.01, 0.0, 0.01, 0.0, *edge],
        "D1": [*edge, 0.0, 0.01, -0.01, 0.0, *edge],
    }
    turnover = {"D10": 0.1, UNIVERSE: 0.02, "D1": 0.05}
    costs = {"D10": 0.002, UNIVERSE: 0.0, "D1": 0.001}
    daily = pd.DataFrame(
        [
            {
                "date": day,
                "portfolio": name,
                "gross": values[k] + costs[name],
                "cost": costs[name],
                "net": values[k],
                "turnover": turnover[name],
                "sleeves": sleeves[k],
            }
            for name, values in nets.items()
            for k, day in enumerate(days)
        ]
    )

    summary = tranche_summary(daily, 2)

    january, february = 0.96 * 1.01 - 1, 1.02 * 1.0 - 1
    top = summary.loc["D10"]
    assert top["monthly_return"] == pytest.approx((january + february) / 2)
    assert top["monthly_excess"] == pytest.approx((january - 0.01 + february - 0.01) / 2)
    assert top["excess_drawdown"] == pytest.approx(-0.05)
    assert top["annual_turnover"] == pytest.approx(0.1 * 252)
    assert top["n_days"] == 4
    spread = summary.loc["D10-D1"]
    paired = [0.958 * 0.998 - 1, 1.028 * 0.998 - 1]
    assert spread["monthly_return"] == pytest.approx(sum(paired) / 2)
    assert spread["monthly_excess"] == pytest.approx(sum(paired) / 2)
    assert spread["excess_drawdown"] == pytest.approx(0.958 * 0.998 - 1)
    assert spread["annual_turnover"] == pytest.approx(0.15 * 252)
    assert summary.loc[UNIVERSE, "monthly_excess"] == pytest.approx(0.0)
    assert summary.loc[UNIVERSE, "excess_drawdown"] == pytest.approx(0.0)


def test_top_liquidity_keeps_the_most_traded_third_of_each_day() -> None:
    frame = pd.DataFrame(
        {
            "date": [DAYS[0]] * 7 + [DAYS[1]] * 3,
            "stock_id": [f"{1000 + k}" for k in range(10)],
            "liq_dollar_vol_med_20d": [6.0, 1.0, 5.0, 2.0, 4.0, 3.0, NAN, 1.0, 9.0, 2.0],
        }
    )

    kept = top_liquidity(frame)

    assert kept["stock_id"].tolist() == ["1000", "1002", "1008"]
    assert list(kept.columns) == list(frame.columns)


def test_a_summary_without_full_days_is_empty_and_nan_scores_get_no_group() -> None:
    bars = prices({"1101": [(10, 10), (10, 11), (11, 12)]})
    frame = signal({"1101": 1.0, "2330": NAN})

    summary = tranche_summary(tranche_returns(frame, bars, CALENDAR, 5, groups=1), 5)
    groups = decile_groups(frame, 1)

    assert summary.empty
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert groups.iloc[0] == 1
    assert math.isnan(groups.iloc[1])
