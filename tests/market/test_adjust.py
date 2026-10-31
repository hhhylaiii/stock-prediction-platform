from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from apps.market.adjust import adjust_prices, adjusted


def test_adjusted_multiplies_price_by_factor() -> None:
    assert adjusted(Decimal("100.5"), 0.5) == pytest.approx(50.25)
    assert adjusted(80.0, 1.0) == pytest.approx(80.0)


def test_adjusted_keeps_missing_price_missing() -> None:
    assert adjusted(None, 0.9) is None


def test_adjust_prices_returns_adjusted_columns_without_mutating() -> None:
    frame = pd.DataFrame(
        {
            "open": [Decimal("10"), Decimal("12")],
            "high": [Decimal("11"), Decimal("13")],
            "low": [Decimal("9"), Decimal("11")],
            "close": [Decimal("10.5"), None],
            "adj_factor": [0.8, 1.0],
        }
    )
    original = frame.copy()
    result = adjust_prices(frame)
    assert list(result.columns) == ["adj_open", "adj_high", "adj_low", "adj_close"]
    assert result.index.equals(frame.index)
    assert result["adj_open"].tolist() == pytest.approx([8.0, 12.0])
    assert result["adj_high"].tolist() == pytest.approx([8.8, 13.0])
    assert result["adj_low"].tolist() == pytest.approx([7.2, 11.0])
    assert result["adj_close"].iloc[0] == pytest.approx(8.4)
    assert np.isnan(result["adj_close"].iloc[1])
    pd.testing.assert_frame_equal(frame, original)


def test_adjust_prices_only_covers_present_columns() -> None:
    frame = pd.DataFrame({"close": [20.0], "adj_factor": [0.5]}, index=[7])
    result = adjust_prices(frame)
    assert list(result.columns) == ["adj_close"]
    assert result.loc[7, "adj_close"] == pytest.approx(10.0)


def test_adjust_prices_requires_factor() -> None:
    with pytest.raises(KeyError, match="adj_factor column is required"):
        adjust_prices(pd.DataFrame({"close": [1.0]}))
