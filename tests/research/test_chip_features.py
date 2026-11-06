import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from apps.market.calendar import TradingDays
from apps.research.features.chip import (
    chip_features,
    holding_changes,
    second_batch_chip_features,
)
from apps.research.features.inputs import FEATURE_WARMUP
from apps.research.features.registry import Group, feature_names
from tests.research.feature_panels import CALENDAR, at, listing, rising

NAN = math.nan


def chips(n: int, **columns: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "fi_net": [1_000.0 * (k + 1) for k in range(n)],
        "it_net": 100.0,
        "dealer_self_net": -50.0,
        "dealer_hedge_net": 20.0,
        "margin_balance": [10_000.0 + 100 * k for k in range(n)],
        "margin_limit": 50_000.0,
        "short_balance": 2_000.0,
        "sbl_short_balance": [5_000.0 + 10 * k for k in range(n)],
        "fi_holding_pct": 30.0,
        "day_trade_volume": 100.0,
        "big_holder_pct_chg": NAN,
        "holder_count_chg": NAN,
    }
    return {**base, **columns}


def test_columns_follow_the_registry() -> None:
    features = chip_features(listing("2330", rising(25)).assign(**chips(25)))

    assert list(features.columns) == ["stock_id", "date", *feature_names(Group.CHIP)]


def test_net_flows_are_summed_over_trading_days_and_divided_by_shares() -> None:
    panel = listing("2330", rising(25), shares=1_000_000.0).assign(**chips(25))

    features = chip_features(panel)

    assert at(features, "2330", 24, "chip_fi_net_1d") == pytest.approx(25_000 / 1e6)
    assert at(features, "2330", 24, "chip_fi_net_5d") == pytest.approx(
        sum(range(21, 26)) * 1_000 / 1e6
    )
    assert at(features, "2330", 24, "chip_fi_net_20d") == pytest.approx(
        sum(range(6, 26)) * 1_000 / 1e6
    )
    assert math.isnan(at(features, "2330", 3, "chip_fi_net_5d"))
    assert at(features, "2330", 24, "chip_it_net_5d") == pytest.approx(500 / 1e6)
    assert at(features, "2330", 24, "chip_dealer_self_net_5d") == pytest.approx(-250 / 1e6)
    assert at(features, "2330", 24, "chip_dealer_hedge_net_5d") == pytest.approx(100 / 1e6)


def test_halted_days_add_no_flow_and_missing_rows_leave_the_window_open() -> None:
    closes = rising(10)
    closes[6] = NAN
    flows = [1_000.0] * 10
    flows[6] = NAN
    flows[8] = NAN
    panel = listing("2330", closes, shares=1_000.0).assign(**chips(10, fi_net=flows))

    features = chip_features(panel)

    assert at(features, "2330", 7, "chip_fi_net_5d") == pytest.approx(4.0)
    assert math.isnan(at(features, "2330", 9, "chip_fi_net_5d"))


def test_margin_and_lending_changes_and_ratios() -> None:
    panel = listing("2330", rising(10), shares=1_000_000.0).assign(**chips(10))

    features = chip_features(panel)

    assert at(features, "2330", 9, "chip_margin_chg_5d") == pytest.approx(500 / 1e6)
    assert at(features, "2330", 9, "chip_sbl_short_chg_5d") == pytest.approx(50 / 1e6)
    assert at(features, "2330", 9, "chip_margin_usage") == pytest.approx(10_900 / 50_000)
    assert at(features, "2330", 9, "chip_short_ratio") == pytest.approx(2_000 / 10_900)
    assert at(features, "2330", 9, "chip_fi_holding_pct") == 30.0
    assert math.isnan(at(features, "2330", 4, "chip_margin_chg_5d"))


def test_balances_carry_over_halted_days() -> None:
    closes = rising(8)
    closes[2] = NAN
    balances = chips(
        8,
        margin_balance=[100.0, 110.0, NAN, 130.0, 140.0, 150.0, 160.0, 170.0],
        margin_limit=[5_000.0, 4_000.0, NAN, *[5_000.0] * 5],
        short_balance=[20.0, 22.0, NAN, *[20.0] * 5],
        fi_holding_pct=[30.0, 31.0, NAN, *[30.0] * 5],
    )
    panel = listing("2330", closes, shares=10.0).assign(**balances)

    features = chip_features(panel)

    assert at(features, "2330", 7, "chip_margin_chg_5d") == pytest.approx((170 - 110) / 10)
    assert at(features, "2330", 2, "chip_margin_usage") == pytest.approx(110 / 4_000)
    assert at(features, "2330", 2, "chip_short_ratio") == pytest.approx(22 / 110)
    assert at(features, "2330", 2, "chip_fi_holding_pct") == 31.0


def test_a_traded_day_without_a_balance_is_not_filled() -> None:
    margin = [100.0, 110.0, NAN, 130.0, 140.0, 150.0, 160.0, 170.0]
    panel = listing("2330", rising(8), shares=10.0).assign(**chips(8, margin_balance=margin))

    features = chip_features(panel)

    assert np.isnan(at(features, "2330", 2, "chip_margin_usage"))
    assert np.isnan(at(features, "2330", 7, "chip_margin_chg_5d"))


def test_zero_limits_and_balances_give_missing_ratios() -> None:
    panel = listing("2330", rising(3)).assign(**chips(3, margin_limit=0.0, margin_balance=0.0))

    features = chip_features(panel)

    assert features["chip_margin_usage"].isna().all()
    assert features["chip_short_ratio"].isna().all()


def holdings(weeks: list[date], scales: list[float]) -> pd.DataFrame:
    rows = []
    for week, scale in zip(weeks, scales, strict=True):
        for level in range(1, 18):
            rows.append(
                {
                    "stock_id": "2330",
                    "data_date": week,
                    "publish_date": week + timedelta(days=1),
                    "level": level,
                    "holders": int(100 * scale) if level <= 15 else int(999 * scale),
                    "pct": 5.0 * scale if level in (11, 12, 13, 14, 15, 16, 17) else 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_weekly_holding_changes_compare_consecutive_weeks() -> None:
    weeks = [date(2016, 3, 4), date(2016, 3, 11), date(2016, 3, 18)]

    changes = holding_changes(holdings(weeks, [1.0, 1.1, 1.21]), CALENDAR)

    assert changes["publish_date"].tolist() == [date(2016, 3, d) for d in (5, 12, 19)]
    assert changes["big_holder_pct_chg"].tolist()[1:] == pytest.approx([2.0, 2.2])
    assert changes["holder_count_chg"].tolist()[1:] == pytest.approx(
        [1_650 / 1_500 - 1, 1_815 / 1_650 - 1]
    )
    assert changes.iloc[0][["big_holder_pct_chg", "holder_count_chg"]].isna().all()


def test_a_missing_week_or_a_reused_code_gives_no_weekly_change() -> None:
    weeks = [date(2012, 6, 1), date(2016, 3, 4), date(2016, 3, 18), date(2016, 3, 25)]

    changes = holding_changes(holdings(weeks, [3.0, 1.0, 1.1, 1.21]), CALENDAR)

    assert changes["big_holder_pct_chg"].isna().tolist() == [True, True, True, False]
    assert changes["holder_count_chg"].isna().tolist() == [True, True, True, False]


def test_a_week_without_trading_does_not_break_the_weekly_change() -> None:
    closed = {date(2016, 2, 4) + timedelta(days=k) for k in range(9)}
    days = [
        day for day in CALENDAR.between(date(2016, 1, 4), date(2016, 3, 31)) if day not in closed
    ]
    calendar = TradingDays(days, date(2016, 1, 1), date(2016, 3, 31))
    weeks = [date(2016, 1, 29), date(2016, 2, 3), date(2016, 2, 19)]

    changes = holding_changes(holdings(weeks, [1.0, 1.1, 1.21]), calendar)

    assert changes["big_holder_pct_chg"].tolist()[1:] == pytest.approx([2.0, 2.2])


def test_unsorted_panel_is_refused() -> None:
    with pytest.raises(ValueError, match="sorted"):
        chip_features(listing("2330", rising(5)).assign(**chips(5)).iloc[::-1])


def test_weekly_changes_pass_through() -> None:
    panel = listing("2330", rising(3)).assign(**chips(3, big_holder_pct_chg=[NAN, 1.5, 1.5]))

    assert at(chip_features(panel), "2330", 1, "chip_big_holder_pct_chg_1w") == 1.5
    assert np.isnan(at(chip_features(panel), "2330", 0, "chip_big_holder_pct_chg_1w"))


def test_later_load_start_gives_the_same_values_across_a_long_halt() -> None:
    start, n = 300, 300 + FEATURE_WARMUP + 140
    closes = rising(n)
    closes[start - 10 : start + 590] = [NAN] * 600
    margin = [NAN if math.isnan(close) else 1_000.0 + k for k, close in enumerate(closes)]
    full_panel = listing("2330", closes, start=date(2012, 1, 2)).assign(
        **chips(n, margin_balance=margin)
    )
    late_panel = full_panel.iloc[start:].reset_index(drop=True)

    full, late = chip_features(full_panel), chip_features(late_panel)

    pd.testing.assert_frame_equal(
        full.iloc[start + FEATURE_WARMUP :].reset_index(drop=True),
        late.iloc[FEATURE_WARMUP:].reset_index(drop=True),
    )


def test_second_batch_columns_follow_the_registry() -> None:
    features = second_batch_chip_features(listing("2330", rising(25)).assign(**chips(25)))

    assert list(features.columns) == ["stock_id", "date", *feature_names(Group.CHIP, batch=2)]


def test_day_trade_ratio_reads_only_days_settled_two_trading_days_ago() -> None:
    closes = rising(12)
    closes[3] = NAN
    volume = [1_000.0 * (k + 1) for k in range(12)]
    day_trade = [100.0 * (k + 1) for k in range(12)]
    day_trade[9] = NAN
    panel = listing("2330", closes, volume=volume).assign(**chips(12, day_trade_volume=day_trade))

    features = second_batch_chip_features(panel)

    traded = [0.0 if k == 3 else v for k, v in enumerate(volume)]
    trades = [0.0 if k == 3 else v for k, v in enumerate(day_trade)]
    assert math.isnan(at(features, "2330", 5, "chip_day_trade_ratio_5d"))
    assert at(features, "2330", 6, "chip_day_trade_ratio_5d") == pytest.approx(
        sum(trades[0:5]) / sum(traded[0:5])
    )
    assert at(features, "2330", 10, "chip_day_trade_ratio_5d") == pytest.approx(
        sum(trades[4:9]) / sum(traded[4:9])
    )
    assert math.isnan(at(features, "2330", 11, "chip_day_trade_ratio_5d"))


def test_a_stock_left_out_of_its_market_day_trade_table_was_not_eligible() -> None:
    eligible = [100.0] * 9
    eligible[5] = NAN
    left_out = [NAN] * 9
    panel = pd.concat(
        [
            listing("2330", rising(9)).assign(**chips(9, day_trade_volume=eligible)),
            listing("2014", rising(9)).assign(**chips(9, day_trade_volume=left_out)),
            listing("6488", rising(9), market="TPEx").assign(**chips(9, day_trade_volume=left_out)),
        ],
        ignore_index=True,
    ).sort_values(["stock_id", "date"], ignore_index=True)

    features = second_batch_chip_features(panel)

    assert at(features, "2014", 6, "chip_day_trade_ratio_5d") == 0.0
    assert math.isnan(at(features, "2014", 8, "chip_day_trade_ratio_5d"))
    assert at(features, "2330", 6, "chip_day_trade_ratio_5d") == pytest.approx(0.1)
    assert math.isnan(at(features, "2330", 8, "chip_day_trade_ratio_5d"))
    assert math.isnan(at(features, "6488", 6, "chip_day_trade_ratio_5d"))


def test_holding_change_institutional_net_and_buy_streak() -> None:
    closes = rising(27)
    closes[6] = NAN
    flows = [1.0, 2.0, -1.0, 3.0, 1.0, 2.0, 5.0, 4.0, NAN, 6.0] + [1.0] * 17
    holding = [NAN if k == 6 else 30.0 + 0.1 * k for k in range(27)]
    panel = listing("2330", closes, shares=1_000_000.0).assign(
        **chips(27, fi_net=flows, fi_holding_pct=holding)
    )

    features = second_batch_chip_features(panel)

    streak = [at(features, "2330", k, "chip_fi_buy_streak") for k in range(10)]
    assert streak[:8] == [1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0]
    assert math.isnan(streak[8])
    assert streak[9] == 1.0
    assert at(features, "2330", 25, "chip_fi_buy_streak") == 17.0
    assert math.isnan(at(features, "2330", 19, "chip_fi_holding_chg_20d"))
    assert at(features, "2330", 25, "chip_fi_holding_chg_20d") == pytest.approx(2.0)
    assert at(features, "2330", 26, "chip_fi_holding_chg_20d") == pytest.approx(2.1)
    assert math.isnan(at(features, "2330", 18, "chip_it_net_20d"))
    assert at(features, "2330", 25, "chip_it_net_20d") == pytest.approx(19 * 100 / 1e6)
