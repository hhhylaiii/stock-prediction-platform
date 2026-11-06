import math
from datetime import date

import pandas as pd
import pytest

from apps.research.features.residual import (
    COVERAGE_FLOOR,
    FALLBACKS,
    MASKED_MONTHS,
    affected_months,
    apply_residual_rules,
    monthly_coverage,
)

NAN = math.nan


def d(text: str) -> date:
    return date.fromisoformat(text)


def frame(rows: list[tuple[str, str, float, float, float, float]]) -> pd.DataFrame:
    columns = ["fund_rev_yoy", "fund_rev_3m_yoy", "fund_rev_mom", "fund_rev_yoy_q"]
    return pd.DataFrame(
        {
            "stock_id": [row[0] for row in rows],
            "date": [d(row[1]) for row in rows],
            **{name: [row[k + 2] for row in rows] for k, name in enumerate(columns)},
        }
    )


def test_monthly_revenue_growth_falls_back_to_the_quarterly_growth_for_every_stock() -> None:
    features = frame(
        [
            ("2330", "2017-03-01", 0.10, 0.20, 0.01, 0.30),
            ("2330", "2017-03-02", NAN, NAN, NAN, 0.30),
            ("1101", "2017-03-02", NAN, NAN, NAN, NAN),
        ]
    )

    ruled = apply_residual_rules(features)

    assert ruled["fund_rev_yoy"].tolist()[:2] == [0.10, 0.30]
    assert ruled["fund_rev_3m_yoy"].tolist()[:2] == [0.20, 0.30]
    assert math.isnan(ruled["fund_rev_yoy"].iloc[2])
    assert math.isnan(ruled["fund_rev_mom"].iloc[1])
    assert ruled["fund_rev_yoy_q"].tolist()[:2] == [0.30, 0.30]
    assert features["fund_rev_yoy"].isna().sum() == 2


def test_month_on_month_revenue_is_masked_for_every_stock_in_the_affected_months() -> None:
    features = frame(
        [
            ("2330", "2017-02-28", 0.1, 0.1, 0.01, 0.1),
            ("2330", "2017-03-01", 0.1, 0.1, 0.02, 0.1),
            ("1101", "2017-03-31", 0.1, 0.1, 0.03, 0.1),
            ("1101", "2017-04-03", 0.1, 0.1, 0.04, 0.1),
        ]
    )

    ruled = apply_residual_rules(features, masked={"fund_rev_mom": frozenset({"2017-03"})})

    mom = ruled["fund_rev_mom"].tolist()
    assert mom[0] == 0.01
    assert math.isnan(mom[1])
    assert math.isnan(mom[2])
    assert mom[3] == 0.04
    assert ruled["fund_rev_yoy"].tolist() == [0.1] * 4


def test_the_rules_are_parameters_with_the_agreed_defaults() -> None:
    assert dict(FALLBACKS) == {
        "fund_rev_yoy": "fund_rev_yoy_q",
        "fund_rev_3m_yoy": "fund_rev_yoy_q",
    }
    assert set(MASKED_MONTHS) == {"fund_rev_mom"}
    assert COVERAGE_FLOOR == 0.9
    features = frame([("2330", "2017-03-01", NAN, 0.2, 0.01, 0.3)])

    untouched = apply_residual_rules(features, fallbacks={}, masked={})

    pd.testing.assert_frame_equal(untouched, features)


def test_monthly_coverage_counts_universe_rows_with_a_value_per_group() -> None:
    rows = pd.DataFrame(
        {
            "stock_id": ["9999", "9999", "2330", "2330", "2330", "1101"],
            "date": [
                d(day)
                for day in ("2017-03-01", "2017-03-02", "2017-03-01", "2017-03-02", "2017-04-03")
            ]
            + [d("2017-03-01")],
            "in_universe": [True, True, True, True, True, False],
            "fund_rev_mom": [0.1, NAN, 0.1, 0.2, NAN, NAN],
        }
    )
    delisted = pd.Series({"9999": True, "2330": False, "1101": False})

    coverage = monthly_coverage(rows, "fund_rev_mom", delisted)

    assert list(coverage.columns) == ["delisted", "current"]
    cell = coverage.to_dict()
    assert cell["delisted"]["2017-03"] == 0.5
    assert cell["current"]["2017-03"] == 1.0
    assert math.isnan(cell["delisted"]["2017-04"])
    assert cell["current"]["2017-04"] == 0.0


def test_affected_months_are_those_where_delisted_coverage_falls_under_the_floor() -> None:
    coverage = pd.DataFrame(
        {"delisted": [0.95, 0.80, 0.50, NAN, 0.0], "current": [1.0, 0.90, 1.0, 1.0, 0.0]},
        index=["2016-01", "2016-02", "2016-03", "2016-04", "2016-05"],
    )

    assert affected_months(coverage) == ["2016-02", "2016-03"]
    assert affected_months(coverage, floor=0.6) == ["2016-03"]
    assert affected_months(coverage, floor=0.4) == []


@pytest.mark.parametrize("floor", [0.0, 1.5])
def test_a_floor_outside_zero_to_one_is_refused(floor: float) -> None:
    coverage = pd.DataFrame({"delisted": [0.5], "current": [1.0]}, index=["2016-01"])

    with pytest.raises(ValueError, match="floor"):
        affected_months(coverage, floor=floor)
