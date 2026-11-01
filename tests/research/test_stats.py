import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from apps.research.metrics.stats import newey_west_t, paired_nw_t


def ar1(n: int, phi: float, rng: np.random.Generator, mean: float = 0.0) -> npt.NDArray[np.float64]:
    noise = rng.standard_normal(n)
    out = np.empty(n)
    out[0] = noise[0] / math.sqrt(1 - phi**2)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out + mean


def test_hand_computed_values() -> None:
    assert newey_west_t([1.0, 2.0, 3.0, 4.0], lag=1) == pytest.approx(4.0)
    assert newey_west_t([1.0, 2.0, 3.0, 4.0], lag=0) == pytest.approx(2.5 / math.sqrt(1.25 / 4))


def test_long_run_variance_matches_bartlett_expectation_on_ar1() -> None:
    phi, lag, n = 0.5, 20, 200_000
    series = ar1(n, phi, np.random.default_rng(7), mean=0.01)
    gamma0 = 1 / (1 - phi**2)
    expected = gamma0 * (1 + 2 * sum((1 - j / (lag + 1)) * phi**j for j in range(1, lag + 1)))
    implied = (series.mean() / newey_west_t(series, lag)) ** 2 * n
    assert implied == pytest.approx(expected, rel=0.05)


def test_autocorrelated_null_series_rejects_near_nominal_rate() -> None:
    rng = np.random.default_rng(11)
    naive, adjusted = [], []
    for _ in range(400):
        series = ar1(2_000, 0.8, rng)
        naive.append(abs(newey_west_t(series, lag=0)) > 1.96)
        adjusted.append(abs(newey_west_t(series, lag=20)) > 1.96)
    assert np.mean(naive) > 0.3
    assert np.mean(adjusted) < 0.15


def test_nan_values_are_dropped() -> None:
    assert newey_west_t([1.0, math.nan, 2.0, 3.0, 4.0], lag=1) == pytest.approx(4.0)


@pytest.mark.parametrize(
    "values", [[], [1.0], [math.nan, 2.0], [3.0, 3.0, 3.0], [0.2, 0.2, 0.2, 0.2]]
)
def test_undefined_t_is_nan(values: list[float]) -> None:
    assert math.isnan(newey_west_t(values, lag=2))


def test_negative_lag_is_refused() -> None:
    with pytest.raises(ValueError, match="lag"):
        newey_west_t([1.0, 2.0], lag=-1)


def test_paired_t_uses_shared_index_and_flips_sign() -> None:
    rng = np.random.default_rng(3)
    index = pd.RangeIndex(500)
    base = pd.Series(rng.standard_normal(500), index=index)
    better = base + 0.2 + 0.1 * pd.Series(rng.standard_normal(500), index=index)
    extra = pd.Series([100.0], index=[9_999])
    forward = paired_nw_t(pd.concat([better, extra]), base, lag=5)
    assert forward > 2
    assert paired_nw_t(base, better, lag=5) == pytest.approx(-forward)
    assert forward == pytest.approx(newey_west_t((better - base).to_numpy(), lag=5))


def test_paired_t_matches_by_label_not_by_position() -> None:
    rng = np.random.default_rng(8)
    first = pd.Series(rng.standard_normal(300) + 0.3, index=pd.RangeIndex(300))
    second = pd.Series(rng.standard_normal(300), index=pd.RangeIndex(300))
    shuffled = second.drop(150).sample(frac=1, random_state=0)
    aligned = (first.drop(150) - second.drop(150)).to_numpy()
    assert paired_nw_t(first, shuffled, lag=3) == pytest.approx(newey_west_t(aligned, lag=3))
