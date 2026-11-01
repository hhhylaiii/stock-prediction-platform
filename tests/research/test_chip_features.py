import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from apps.research.features.chip import chip_features, holding_changes
from apps.research.features.registry import Group, feature_names
from tests.research.feature_panels import at, listing, rising

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
    margin = [100.0, 110.0, NAN, 130.0, 140.0, 150.0, 160.0, 170.0]
    panel = listing("2330", closes, shares=10.0).assign(**chips(8, margin_balance=margin))

    features = chip_features(panel)

    assert at(features, "2330", 7, "chip_margin_chg_5d") == pytest.approx((170 - 110) / 10)


def test_zero_limits_and_balances_give_missing_ratios() -> None:
    panel = listing("2330", rising(3)).assign(**chips(3, margin_limit=0.0, margin_balance=0.0))

    features = chip_features(panel)

    assert features["chip_margin_usage"].isna().all()
    assert features["chip_short_ratio"].isna().all()


def test_weekly_holding_changes_compare_consecutive_weeks() -> None:
    weeks = [date(2016, 3, 4), date(2016, 3, 11), date(2016, 3, 18)]
    rows = []
    for week, scale in zip(weeks, (1.0, 1.1, 1.21), strict=True):
        for level in range(1, 18):
            rows.append(
                {
                    "stock_id": "2330",
                    "data_date": week,
                    "publish_date": date.fromordinal(week.toordinal() + 1),
                    "level": level,
                    "holders": int(100 * scale) if level <= 15 else 999,
                    "pct": 5.0 * scale if level in (12, 13, 14, 15) else 1.0,
                }
            )

    changes = holding_changes(pd.DataFrame(rows))

    assert changes["publish_date"].tolist() == [date(2016, 3, 12), date(2016, 3, 19)]
    assert changes["big_holder_pct_chg"].tolist() == pytest.approx([2.0, 2.2])
    assert changes["holder_count_chg"].tolist() == pytest.approx(
        [1_650 / 1_500 - 1, 1_815 / 1_650 - 1]
    )


def test_unsorted_panel_is_refused() -> None:
    with pytest.raises(ValueError, match="sorted"):
        chip_features(listing("2330", rising(5)).assign(**chips(5)).iloc[::-1])


def test_weekly_changes_pass_through() -> None:
    panel = listing("2330", rising(3)).assign(**chips(3, big_holder_pct_chg=[NAN, 1.5, 1.5]))

    assert at(chip_features(panel), "2330", 1, "chip_big_holder_pct_chg_1w") == 1.5
    assert np.isnan(at(chip_features(panel), "2330", 0, "chip_big_holder_pct_chg_1w"))
