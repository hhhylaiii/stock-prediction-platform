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


def test_adjust_prices_adds_adjusted_columns_without_mutating() -> None:
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
    assert result["adj_open"].tolist() == pytest.approx([8.0, 12.0])
    assert result["adj_high"].tolist() == pytest.approx([8.8, 13.0])
    assert result["adj_low"].tolist() == pytest.approx([7.2, 11.0])
    assert result["adj_close"].iloc[0] == pytest.approx(8.4)
    assert np.isnan(result["adj_close"].iloc[1])
    pd.testing.assert_frame_equal(frame, original)


def test_adjust_prices_only_touches_present_columns() -> None:
    result = adjust_prices(pd.DataFrame({"close": [20.0], "adj_factor": [0.5]}))
    assert list(result.columns) == ["close", "adj_factor", "adj_close"]


def test_adjust_prices_requires_factor() -> None:
    with pytest.raises(KeyError, match="adj_factor"):
        adjust_prices(pd.DataFrame({"close": [1.0]}))
