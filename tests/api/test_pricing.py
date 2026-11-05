import math

import pytest

from apps.api.pricing import (
    QUANTILE,
    TECH_VOL,
    VOLATILITY,
    choose_interval,
    price_at,
    quantile_interval,
    volatility_interval,
)


def spread_of(vol: float, horizon: int) -> float:
    return 1.28 * vol * math.sqrt(horizon / 252)


def test_predicted_price_is_close_times_one_plus_return() -> None:
    assert price_at(1085.0, 0.042) == pytest.approx(1085.0 * 1.042)
    assert price_at(50.0, -0.1) == pytest.approx(45.0)


def test_quantile_interval_converts_both_ends_on_the_close() -> None:
    interval = quantile_interval(200.0, -0.05, 0.12)

    assert interval.low == pytest.approx(190.0)
    assert interval.high == pytest.approx(224.0)
    assert (interval.ret_low, interval.ret_high, interval.source) == (-0.05, 0.12, QUANTILE)


def test_volatility_interval_is_centred_on_the_predicted_return() -> None:
    spread = spread_of(0.4, 20)

    interval = volatility_interval(100.0, 0.15, 0.4, 20)

    assert interval.ret_low == pytest.approx(0.15 - spread)
    assert interval.ret_high == pytest.approx(0.15 + spread)
    assert interval.low == pytest.approx(100.0 * (1.15 - spread))
    assert interval.high == pytest.approx(100.0 * (1.15 + spread))
    assert interval.low < price_at(100.0, 0.15) < interval.high
    assert interval.source == VOLATILITY


def test_volatility_interval_narrows_with_a_shorter_horizon() -> None:
    five = volatility_interval(100.0, 0.0, 0.4, 5)
    twenty = volatility_interval(100.0, 0.0, 0.4, 20)

    assert twenty.low < five.low < 100.0 < five.high < twenty.high


def test_a_wide_band_never_prices_below_zero() -> None:
    interval = volatility_interval(10.0, -0.2, 4.0, 20)

    assert (interval.ret_low, interval.low) == (-1.0, 0.0)
    assert interval.ret_high == pytest.approx(-0.2 + spread_of(4.0, 20))


def test_quantiles_win_unless_their_model_is_on_a_coverage_alert() -> None:
    usual = choose_interval(
        100.0, 20, ret=0.04, q10=-0.1, q90=0.1, vol=0.3, tech_vol=0.5, quantile_alert=False
    )
    alerted = choose_interval(
        100.0, 20, ret=0.04, q10=-0.1, q90=0.1, vol=0.3, tech_vol=0.5, quantile_alert=True
    )

    assert usual is not None
    assert usual.source == QUANTILE
    assert alerted is not None
    assert alerted.source == VOLATILITY
    assert alerted.ret_low == pytest.approx(0.04 - spread_of(0.3, 20))
    assert alerted.low <= price_at(100.0, 0.04) <= alerted.high


@pytest.mark.parametrize(
    ("q10", "q90", "vol", "tech_vol", "expected"),
    [
        (None, None, 0.3, 0.5, VOLATILITY),
        (-0.1, None, 0.3, None, VOLATILITY),
        (None, None, None, 0.5, TECH_VOL),
        (None, None, None, None, None),
    ],
)
def test_missing_inputs_fall_back_to_volatility_then_realised_volatility(
    q10: float | None,
    q90: float | None,
    vol: float | None,
    tech_vol: float | None,
    expected: str | None,
) -> None:
    interval = choose_interval(
        100.0, 20, ret=None, q10=q10, q90=q90, vol=vol, tech_vol=tech_vol, quantile_alert=False
    )

    assert (interval.source if interval else None) == expected


def test_the_realised_fallback_uses_tech_vol_around_the_return() -> None:
    interval = choose_interval(
        100.0, 5, ret=0.02, q10=None, q90=None, vol=None, tech_vol=0.5, quantile_alert=False
    )

    assert interval is not None
    assert interval.ret_high == pytest.approx(0.02 + spread_of(0.5, 5))


def test_alerted_quantiles_without_any_volatility_give_no_interval() -> None:
    interval = choose_interval(
        100.0, 20, ret=0.0, q10=-0.1, q90=0.1, vol=None, tech_vol=None, quantile_alert=True
    )

    assert interval is None
