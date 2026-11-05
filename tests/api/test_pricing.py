import math

import pytest

from apps.api.pricing import (
    QUANTILE,
    VOLATILITY,
    Interval,
    choose_interval,
    price_at,
    quantile_interval,
    volatility_interval,
)


def test_predicted_price_is_close_times_one_plus_return() -> None:
    assert price_at(1085.0, 0.042) == pytest.approx(1085.0 * 1.042)
    assert price_at(50.0, -0.1) == pytest.approx(45.0)


def test_quantile_interval_converts_both_ends_on_the_close() -> None:
    interval = quantile_interval(200.0, -0.05, 0.12)

    assert interval == Interval(
        low=pytest.approx(190.0),
        high=pytest.approx(224.0),
        ret_low=-0.05,
        ret_high=0.12,
        source=QUANTILE,
    )


def test_volatility_interval_is_1_28_sigma_over_the_horizon() -> None:
    spread = 1.28 * 0.4 * math.sqrt(20 / 252)

    interval = volatility_interval(100.0, 0.4, 20)

    assert interval.low == pytest.approx(100.0 * math.exp(-spread))
    assert interval.high == pytest.approx(100.0 * math.exp(spread))
    assert interval.ret_low == pytest.approx(math.exp(-spread) - 1)
    assert interval.ret_high == pytest.approx(math.exp(spread) - 1)
    assert interval.source == VOLATILITY


def test_volatility_interval_narrows_with_a_shorter_horizon() -> None:
    five = volatility_interval(100.0, 0.4, 5)
    twenty = volatility_interval(100.0, 0.4, 20)

    assert twenty.low < five.low < 100.0 < five.high < twenty.high


def test_quantiles_win_unless_their_model_is_on_a_coverage_alert() -> None:
    usual = choose_interval(100.0, 20, q10=-0.1, q90=0.1, vol=0.3, quantile_alert=False)
    alerted = choose_interval(100.0, 20, q10=-0.1, q90=0.1, vol=0.3, quantile_alert=True)

    assert [usual.source if usual else None, alerted.source if alerted else None] == [
        QUANTILE,
        VOLATILITY,
    ]


@pytest.mark.parametrize(
    ("q10", "q90", "vol", "expected"),
    [
        (None, None, 0.3, VOLATILITY),
        (-0.1, None, 0.3, VOLATILITY),
        (None, None, None, None),
    ],
)
def test_missing_inputs_fall_back_to_volatility_or_nothing(
    q10: float | None, q90: float | None, vol: float | None, expected: str | None
) -> None:
    interval = choose_interval(100.0, 20, q10=q10, q90=q90, vol=vol, quantile_alert=False)

    assert (interval.source if interval else None) == expected


def test_alerted_quantiles_without_a_volatility_give_no_interval() -> None:
    assert choose_interval(100.0, 20, q10=-0.1, q90=0.1, vol=None, quantile_alert=True) is None
