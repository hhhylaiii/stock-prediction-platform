"""Newey-West t statistics for overlapping-label metric series (SPEC §7 評估指標)."""

import math

import numpy as np
import numpy.typing as npt
import pandas as pd

FLAT_TOLERANCE = 1e-12


def is_flat(spread: float, level: float) -> bool:
    """A spread this small next to the level is float noise around a constant series."""
    return spread <= FLAT_TOLERANCE * max(1.0, abs(level))


def newey_west_t(values: npt.ArrayLike, lag: int) -> float:
    """t of the mean with a Bartlett-kernel long-run variance; NaN when undefined."""
    if lag < 0:
        raise ValueError("lag must be zero or positive")
    series = np.asarray(values, dtype=np.float64)
    series = series[~np.isnan(series)]
    n = series.size
    if n < 2:
        return math.nan
    mean = float(series.mean())
    errors = series - mean
    variance = float(errors @ errors) / n
    for j in range(1, min(lag, n - 1) + 1):
        variance += 2 * (1 - j / (lag + 1)) * float(errors[j:] @ errors[:-j]) / n
    if variance <= 0 or is_flat(math.sqrt(variance), mean):
        return math.nan
    return mean / math.sqrt(variance / n)


def paired_nw_t(first: pd.Series, second: pd.Series, lag: int) -> float:
    """Newey-West t of first − second over the index labels both series hold."""
    joined = pd.concat({"first": first, "second": second}, axis=1, join="inner").dropna()
    return newey_west_t((joined["first"] - joined["second"]).to_numpy(), lag)
