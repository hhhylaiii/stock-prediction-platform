import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from apps.research.features.inputs import FEATURE_WARMUP
from apps.research.features.registry import Group, feature_names
from apps.research.features.size_risk import size_liquidity_features
from apps.research.features.technical import (
    second_batch_technical_features,
    technical_features,
)
from tests.research.feature_panels import at, listing, rising

NAN = math.nan
ROOT = math.sqrt(252)


def wavy(n: int) -> list[float]:
    return [100.0 * (1 + 0.03 * math.sin(k * 1.3) + 0.002 * k) for k in range(n)]


def ema(values: list[float], span: int) -> list[float]:
    alpha, result = 2 / (span + 1), [values[0]]
    for value in values[1:]:
        result.append(result[-1] + alpha * (value - result[-1]))
    return result


def wilder_rsi(closes: list[float], period: int = 14) -> list[float]:
    result, gain, loss = [NAN] * len(closes), NAN, NAN
    for t in range(1, len(closes)):
        up, down = max(closes[t] - closes[t - 1], 0.0), max(closes[t - 1] - closes[t], 0.0)
        gain = up if t == 1 else gain + (up - gain) / period
        loss = down if t == 1 else loss + (down - loss) / period
        if t >= period:
            result[t] = 100 * gain / (gain + loss)
    return result


def stochastic(
    highs: list[float], lows: list[float], closes: list[float]
) -> tuple[list[float], list[float]]:
    k_line, d_line, k, d = [NAN] * len(closes), [NAN] * len(closes), NAN, NAN
    for t in range(8, len(closes)):
        high, low = max(highs[t - 8 : t + 1]), min(lows[t - 8 : t + 1])
        rsv = 100 * (closes[t] - low) / (high - low)
        k = rsv if t == 8 else k + (rsv - k) / 3
        d = k if t == 8 else d + (k - d) / 3
        k_line[t], d_line[t] = k, d
    return k_line, d_line


def test_columns_follow_the_registry_and_rows_keep_the_panel_order() -> None:
    panel = pd.concat([listing("2330", rising(30)), listing("1101", rising(5))], ignore_index=True)

    features = technical_features(panel)

    assert list(features.columns) == ["stock_id", "date", *feature_names(Group.TECHNICAL)]
    assert features.index.equals(panel.index)
    assert features["stock_id"].tolist() == panel["stock_id"].tolist()


def test_returns_and_momentum_count_rows_of_the_listing() -> None:
    features = technical_features(listing("2330", rising(300)))

    assert at(features, "2330", 5, "tech_ret_1d") == pytest.approx(105 / 104 - 1)
    assert at(features, "2330", 5, "tech_ret_5d") == pytest.approx(105 / 100 - 1)
    assert math.isnan(at(features, "2330", 19, "tech_ret_20d"))
    assert at(features, "2330", 60, "tech_ret_60d") == pytest.approx(160 / 100 - 1)
    assert math.isnan(at(features, "2330", 249, "tech_mom_250_20"))
    assert at(features, "2330", 250, "tech_mom_250_20") == pytest.approx(330 / 100 - 1)
    assert at(features, "2330", 260, "tech_mom_250_20") == pytest.approx(340 / 110 - 1)


def test_halted_days_carry_the_last_close() -> None:
    closes = rising(10)
    closes[6] = NAN
    features = technical_features(listing("2330", closes))

    assert at(features, "2330", 6, "tech_ret_1d") == pytest.approx(0.0)
    assert at(features, "2330", 7, "tech_ret_1d") == pytest.approx(107 / 105 - 1)


def test_moving_average_gap_needs_a_full_window() -> None:
    features = technical_features(listing("2330", rising(70)))

    assert math.isnan(at(features, "2330", 3, "tech_ma_gap_5d"))
    assert at(features, "2330", 4, "tech_ma_gap_5d") == pytest.approx(104 / 102 - 1)
    assert at(features, "2330", 59, "tech_ma_gap_60d") == pytest.approx(159 / 129.5 - 1)


def test_volatility_is_the_annualized_std_with_halted_days_as_flat() -> None:
    closes = wavy(70)
    closes[50] = NAN
    features = technical_features(listing("2330", closes))
    filled = pd.Series(closes).ffill()
    steps = (filled / filled.shift(1) - 1).to_numpy()

    assert math.isnan(at(features, "2330", 19, "tech_vol_20d"))
    assert at(features, "2330", 20, "tech_vol_20d") == pytest.approx(
        np.std(steps[1:21], ddof=1) * ROOT
    )
    assert steps[50] == 0.0
    assert at(features, "2330", 60, "tech_vol_20d") == pytest.approx(
        np.std(steps[41:61], ddof=1) * ROOT
    )
    assert at(features, "2330", 60, "tech_vol_60d") == pytest.approx(
        np.std(steps[1:61], ddof=1) * ROOT
    )


def test_volume_ratio_and_turnover_rate_count_halted_days_as_zero() -> None:
    closes = [100.0] * 25
    closes[22] = NAN
    volume = [float(1_000 * (k + 1)) for k in range(25)]
    features = technical_features(
        listing("2330", closes, volume=volume, shares=2_000_000.0).assign(
            volume=lambda frame: frame["volume"].where(frame["has_bar"])
        )
    )
    traded = [0.0 if k == 22 else v for k, v in enumerate(volume)]

    five, twenty = np.mean(traded[20:25]), np.mean(traded[5:25])
    assert at(features, "2330", 24, "tech_volume_ratio_5_20") == pytest.approx(five / twenty)
    assert at(features, "2330", 24, "tech_turnover_rate_5d") == pytest.approx(five / 2_000_000)
    assert math.isnan(at(features, "2330", 18, "tech_volume_ratio_5_20"))
    assert math.isnan(at(features, "2330", 3, "tech_turnover_rate_5d"))


def test_rsi_matches_wilder_smoothing() -> None:
    closes = wavy(60)
    expected = wilder_rsi(closes)

    features = technical_features(listing("2330", closes))
    rsi = features["tech_rsi_14"].tolist()

    assert all(math.isnan(value) for value in rsi[:14])
    assert rsi[14:] == pytest.approx(expected[14:])
    assert at(technical_features(listing("1101", rising(20))), "1101", 19, "tech_rsi_14") == 100.0


def test_kd_matches_the_nine_three_three_stochastic() -> None:
    closes = wavy(40)
    highs = [close * (1.01 + 0.004 * (k % 3)) for k, close in enumerate(closes)]
    lows = [close * (0.99 - 0.003 * (k % 4)) for k, close in enumerate(closes)]
    expected_k, expected_d = stochastic(highs, lows, closes)

    features = technical_features(listing("2330", closes, adj_high=highs, adj_low=lows))

    assert features["tech_kd_k"].isna().tolist()[:8] == [True] * 8
    assert features["tech_kd_k"].tolist()[8:] == pytest.approx(expected_k[8:])
    assert features["tech_kd_d"].tolist()[8:] == pytest.approx(expected_d[8:])


def test_macd_histogram_is_divided_by_the_close() -> None:
    closes = wavy(60)
    line = np.array(ema(closes, 12)) - np.array(ema(closes, 26))
    histogram = line - np.array(ema(line.tolist(), 9))

    features = technical_features(listing("2330", closes))
    values = features["tech_macd_hist_pct"].to_numpy()

    assert np.isnan(values[:33]).all()
    assert values[33:] == pytest.approx(histogram[33:] / np.array(closes[33:]))


def test_52_week_gaps_use_the_listed_range_until_a_year_passes() -> None:
    closes = rising(300)
    highs = [close + 1 for close in closes]
    lows = [close - 1 for close in closes]
    features = technical_features(listing("2330", closes, adj_high=highs, adj_low=lows))

    assert at(features, "2330", 2, "tech_low_52w_gap") == pytest.approx(102 / 99 - 1)
    assert at(features, "2330", 2, "tech_high_52w_gap") == pytest.approx(102 / 103 - 1)
    assert at(features, "2330", 260, "tech_low_52w_gap") == pytest.approx(360 / 110 - 1)


def test_52_week_high_drops_out_after_250_rows() -> None:
    closes = [100.0 + k for k in range(20)] + [119.0 - 0.1 * k for k in range(300)]
    features = technical_features(listing("2330", closes))

    assert at(features, "2330", 268, "tech_high_52w_gap") == pytest.approx(closes[268] / 119.0 - 1)
    assert at(features, "2330", 270, "tech_high_52w_gap") == pytest.approx(
        closes[270] / closes[21] - 1
    )


def test_limit_counts_and_attention() -> None:
    limit_up = [k in (3, 10, 25) for k in range(30)]
    limit_down = [k == 12 for k in range(30)]
    attention = [k == 29 for k in range(30)]
    panel = listing(
        "2330",
        rising(30),
        limit_up_close=limit_up,
        limit_down_close=limit_down,
        attention=attention,
    )

    features = technical_features(panel)

    assert at(features, "2330", 25, "tech_limit_up_t") == 1.0
    assert at(features, "2330", 24, "tech_limit_up_t") == 0.0
    assert at(features, "2330", 22, "tech_limit_up_cnt_20d") == 2.0
    assert at(features, "2330", 29, "tech_limit_up_cnt_20d") == 2.0
    assert at(features, "2330", 29, "tech_limit_down_cnt_20d") == 1.0
    assert at(features, "2330", 29, "tech_attention_t") == 1.0


def test_windows_stay_inside_one_listing() -> None:
    old = listing("5450", rising(30), segment=1)
    new = listing("5450", [500.0] * 30, start=pd.Timestamp("2015-06-01").date(), segment=2)
    features = technical_features(pd.concat([old, new], ignore_index=True))
    second = features.iloc[30:].reset_index(drop=True)

    assert math.isnan(second["tech_ret_5d"].iloc[3])
    assert second["tech_ret_5d"].iloc[5] == pytest.approx(0.0)


def test_later_load_start_gives_the_same_values_across_a_long_halt() -> None:
    start = 300
    closes = wavy(start + FEATURE_WARMUP + 100)
    closes[start - 10 : start + 290] = [NAN] * 300
    full_panel = listing("2330", closes, start=date(2012, 1, 2))
    late_panel = listing("2330", closes[start:], start=full_panel["date"][start])

    for build in (technical_features, size_liquidity_features):
        full, late = build(full_panel), build(late_panel)
        pd.testing.assert_frame_equal(
            full.iloc[start + FEATURE_WARMUP :].reset_index(drop=True),
            late.iloc[FEATURE_WARMUP:].reset_index(drop=True),
            rtol=1e-6,
        )


def test_limit_counts_cover_exactly_twenty_rows() -> None:
    limit_up = [k in (9, 10) for k in range(30)]

    features = technical_features(listing("2330", rising(30), limit_up_close=limit_up))

    assert at(features, "2330", 29, "tech_limit_up_cnt_20d") == 1.0
    assert at(features, "2330", 28, "tech_limit_up_cnt_20d") == 2.0


def test_unsorted_panel_is_refused() -> None:
    with pytest.raises(ValueError, match="sorted"):
        technical_features(listing("2330", rising(10)).iloc[::-1])


def test_second_batch_columns_follow_the_registry() -> None:
    features = second_batch_technical_features(listing("2330", rising(30), adj_open=rising(30)))

    assert list(features.columns) == ["stock_id", "date", *feature_names(Group.TECHNICAL, batch=2)]


@pytest.mark.parametrize("bar", [False, True], ids=["halted", "bar without trades"])
def test_parkinson_volatility_averages_squared_log_ranges_with_idle_days_flat(bar: bool) -> None:
    closes = wavy(30)
    closes[25] = NAN
    highs = [c * 1.02 if not math.isnan(c) else NAN for c in closes]
    lows = [c * 0.99 if not math.isnan(c) else NAN for c in closes]
    has_bar = [bar or not math.isnan(c) for c in closes]
    features = second_batch_technical_features(
        listing("2330", closes, adj_high=highs, adj_low=lows, adj_open=closes, has_bar=has_bar)
    )
    squared = [0.0 if k == 25 else math.log(1.02 / 0.99) ** 2 for k in range(30)]

    assert math.isnan(at(features, "2330", 18, "tech_parkinson_vol_20d"))
    assert at(features, "2330", 29, "tech_parkinson_vol_20d") == pytest.approx(
        math.sqrt(np.mean(squared[10:30]) / (4 * math.log(2))) * ROOT
    )


def test_gap_compares_the_open_with_the_carried_previous_close() -> None:
    closes = rising(8)
    closes[4] = NAN
    opens = [c + 0.5 for c in rising(8)]
    features = second_batch_technical_features(listing("2330", closes, adj_open=opens))

    assert math.isnan(at(features, "2330", 0, "tech_gap_1d"))
    assert at(features, "2330", 3, "tech_gap_1d") == pytest.approx(103.5 / 102 - 1)
    assert math.isnan(at(features, "2330", 4, "tech_gap_1d"))
    assert at(features, "2330", 5, "tech_gap_1d") == pytest.approx(105.5 / 103 - 1)


def test_close_location_sits_inside_the_five_day_range() -> None:
    closes = [100.0, 104.0, 98.0, 101.0, 103.0, 102.0, 102.0]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    flat = [100.0] * 7
    panel = pd.concat(
        [
            listing("2330", closes, adj_high=highs, adj_low=lows, adj_open=closes),
            listing("1101", flat, adj_open=flat),
        ],
        ignore_index=True,
    )

    features = second_batch_technical_features(panel)

    assert math.isnan(at(features, "2330", 3, "tech_close_loc_5d"))
    assert at(features, "2330", 4, "tech_close_loc_5d") == pytest.approx((103 - 97) / (105 - 97))
    assert at(features, "2330", 6, "tech_close_loc_5d") == pytest.approx((102 - 97) / (104 - 97))
    assert math.isnan(at(features, "1101", 6, "tech_close_loc_5d"))


def test_max_return_and_skew_need_full_windows_of_flat_returns() -> None:
    closes = wavy(70)
    closes[40] = NAN
    features = second_batch_technical_features(listing("2330", closes, adj_open=closes))
    filled = pd.Series(closes).ffill()
    steps = filled / filled.shift(1) - 1

    assert math.isnan(at(features, "2330", 19, "tech_max_ret_20d"))
    assert at(features, "2330", 20, "tech_max_ret_20d") == pytest.approx(steps[1:21].max())
    assert at(features, "2330", 50, "tech_max_ret_20d") == pytest.approx(steps[31:51].max())
    assert math.isnan(at(features, "2330", 59, "tech_skew_60d"))
    assert at(features, "2330", 60, "tech_skew_60d") == pytest.approx(steps[1:61].skew())
    assert at(features, "2330", 69, "tech_skew_60d") == pytest.approx(steps[10:70].skew())
