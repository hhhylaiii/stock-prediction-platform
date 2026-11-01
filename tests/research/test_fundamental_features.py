import math
from datetime import date, timedelta

import pandas as pd
import pytest

from apps.research.features.fundamental import (
    fundamental_features,
    revenue_signals,
    statement_signals,
)
from apps.research.features.registry import Group, feature_names
from tests.research.feature_panels import at, listing, rising

NAN = math.nan


def months(first: str, values: list[float]) -> pd.DataFrame:
    year, month = map(int, first.split("-"))
    rows = []
    for k, value in enumerate(values):
        y, m = divmod(year * 12 + month - 1 + k, 12)
        rows.append(
            {
                "stock_id": "2330",
                "period": f"{y}-{m + 1:02d}",
                "revenue": value,
                "announce_date": date(y, m + 1, 10) + timedelta(days=31),
            }
        )
    return pd.DataFrame(rows)


def quarters(first: str, n: int, **columns: list[float]) -> pd.DataFrame:
    year, quarter = int(first[:4]), int(first[-1])
    rows = []
    for k in range(n):
        y, q = divmod(year * 4 + quarter - 1 + k, 4)
        row: dict[str, object] = {
            "stock_id": "2330",
            "period": f"{y}-Q{q + 1}",
            "fin_type": "consolidated",
            "eps": 1.0,
            "gross_margin": 0.4,
            "op_margin": 0.2,
            "net_income": 100.0,
            "roe": 0.05,
            "debt_ratio": 0.3,
            "operating_cf": 120.0,
            "announce_date": date(y, q * 3 + 3, 28) + timedelta(days=45),
        }
        rows.append({**row, **{name: values[k] for name, values in columns.items()}})
    return pd.DataFrame(rows)


def one(frame: pd.DataFrame, period: str) -> pd.Series:
    picked = frame[frame["period"] == period]
    assert picked["signal"].is_unique
    return picked.set_index("signal")["value"]


def known(frame: pd.DataFrame, period: str, signal: str) -> pd.Timestamp:
    picked = frame[(frame["period"] == period) & (frame["signal"] == signal)]
    assert len(picked) == 1
    return pd.Timestamp(picked.iloc[0]["known_from"])


def test_revenue_growth_compares_the_same_months_a_year_apart() -> None:
    revenue = months("2015-01", [100.0 + 10 * k for k in range(15)])

    signals = revenue_signals(revenue)
    march = one(signals, "2016-03")

    assert march["fund_rev_yoy"] == pytest.approx(240 / 120 - 1)
    assert march["fund_rev_mom"] == pytest.approx(240 / 230 - 1)
    assert march["fund_rev_3m_yoy"] == pytest.approx((220 + 230 + 240) / (100 + 110 + 120) - 1)
    assert math.isnan(one(signals, "2016-01")["fund_rev_3m_yoy"])
    assert one(signals, "2016-01")["fund_rev_yoy"] == pytest.approx(220 / 100 - 1)


def test_revenue_growth_needs_a_positive_base_and_a_complete_history() -> None:
    revenue = months("2015-01", [0.0] + [100.0] * 13)
    holed = revenue[revenue["period"] != "2015-12"]

    assert math.isnan(one(revenue_signals(revenue), "2016-01")["fund_rev_yoy"])
    assert math.isnan(one(revenue_signals(holed), "2016-01")["fund_rev_mom"])
    assert one(revenue_signals(holed), "2016-02")["fund_rev_mom"] == pytest.approx(0.0)


def test_each_revenue_signal_waits_only_for_the_months_it_reads() -> None:
    revenue = months("2015-01", [100.0] * 16)
    late = pd.Timestamp("2016-06-01")
    revenue.loc[revenue["period"] == "2015-02", "announce_date"] = late.date()

    signals = revenue_signals(revenue)

    assert known(signals, "2016-02", "fund_rev_yoy") == late
    assert known(signals, "2016-02", "fund_rev_mom") < late
    assert known(signals, "2016-03", "fund_rev_3m_yoy") == late
    assert known(signals, "2016-04", "fund_rev_3m_yoy") == late
    assert known(signals, "2016-03", "fund_rev_yoy") < late
    assert known(signals, "2015-03", "fund_rev_mom") == late


def test_ttm_eps_growth_and_ratios_over_four_quarters() -> None:
    statements = quarters(
        "2014-Q1",
        8,
        eps=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.5],
        net_income=[100.0] * 4 + [200.0, 200.0, 100.0, 150.0],
        operating_cf=[50.0] * 8,
        gross_margin=[0.30, 0.31, 0.32, 0.33, 0.40, 0.41, 0.42, 0.43],
        debt_ratio=[0.5] * 7 + [0.45],
    )

    last = one(statement_signals(statements), "2015-Q4")

    assert last["eps_ttm"] == pytest.approx(6.5)
    assert last["fund_eps_yoy"] == pytest.approx(6.5 / 4.0 - 1)
    assert last["fund_roe_ttm"] == pytest.approx(0.2)
    assert last["fund_cfo_to_ni"] == pytest.approx(200 / 650)
    assert last["fund_gross_margin"] == pytest.approx(0.43)
    assert last["fund_gm_yoy_chg"] == pytest.approx(0.43 - 0.33)
    assert last["fund_opm_yoy_chg"] == pytest.approx(0.0)
    assert last["fund_debt_ratio"] == pytest.approx(0.45)
    assert math.isnan(one(statement_signals(statements), "2014-Q4")["fund_eps_yoy"])


@pytest.mark.parametrize("first", [-1.0, -1.25])
def test_eps_growth_is_missing_when_last_years_ttm_is_not_positive(first: float) -> None:
    statements = quarters("2014-Q1", 8, eps=[first, 0.5, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0])

    last = one(statement_signals(statements), "2015-Q4")

    assert last["eps_ttm"] == pytest.approx(4.0)
    assert math.isnan(last["fund_eps_yoy"])


def test_ttm_needs_four_consecutive_quarters_and_positive_income() -> None:
    statements = quarters("2015-Q1", 5, net_income=[100.0, -300.0, 100.0, 50.0, 100.0])
    gap = statements[statements["period"] != "2015-Q2"]

    full = statement_signals(statements)
    holed = statement_signals(gap)

    assert math.isnan(one(full, "2015-Q4")["fund_cfo_to_ni"])
    assert math.isnan(one(holed, "2015-Q4")["eps_ttm"])
    assert math.isnan(one(holed, "2016-Q1")["eps_ttm"])
    assert one(holed, "2015-Q4")["fund_gross_margin"] == pytest.approx(0.4)


def test_consolidated_statements_win_over_individual_ones() -> None:
    consolidated = quarters("2015-Q1", 4)
    individual = quarters("2015-Q1", 4, eps=[9.0] * 4).assign(fin_type="individual")
    only_individual = quarters("2016-Q1", 1, eps=[3.0]).assign(fin_type="individual")

    signals = statement_signals(pd.concat([consolidated, individual, only_individual]))

    assert one(signals, "2015-Q4")["eps_ttm"] == pytest.approx(4.0)
    assert one(signals, "2016-Q1")["eps_ttm"] == pytest.approx(6.0)


def test_features_divide_ttm_eps_by_the_carried_raw_close() -> None:
    closes = rising(4)
    closes[2] = NAN
    panel = listing("2330", closes, adj_close=[c / 2 for c in closes]).assign(
        eps_ttm=[6.0, 6.0, 6.0, -3.0],
        pb=[2.0, 4.0, NAN, 0.5],
        dividend_yield=0.03,
        fund_rev_yoy=0.1,
        fund_rev_3m_yoy=0.2,
        fund_rev_mom=0.3,
        fund_eps_yoy=0.4,
        fund_roe_ttm=0.15,
        fund_gross_margin=0.4,
        fund_op_margin=0.2,
        fund_debt_ratio=0.3,
        fund_gm_yoy_chg=0.01,
        fund_opm_yoy_chg=0.02,
        fund_cfo_to_ni=1.2,
        fund_days_since_fs=[1.0, 2.0, 3.0, 4.0],
    )

    features = fundamental_features(panel)

    assert list(features.columns) == ["stock_id", "date", *feature_names(Group.FUNDAMENTAL)]
    assert at(features, "2330", 1, "fund_ep") == pytest.approx(6.0 / 101.0)
    assert at(features, "2330", 2, "fund_ep") == pytest.approx(6.0 / 101.0)
    assert at(features, "2330", 1, "fund_bp") == pytest.approx(0.25)
    assert math.isnan(at(features, "2330", 2, "fund_bp"))
    assert at(features, "2330", 3, "fund_ep") == pytest.approx(-3.0 / 103.0)
    assert at(features, "2330", 3, "fund_days_since_fs") == 4.0


def test_each_statement_signal_waits_only_for_the_quarters_it_reads() -> None:
    statements = quarters("2014-Q1", 8)
    late = pd.Timestamp("2016-06-01")
    statements.loc[statements["period"] == "2014-Q1", "announce_date"] = late.date()

    signals = statement_signals(statements)

    assert known(signals, "2014-Q4", "eps_ttm") == late
    assert known(signals, "2014-Q4", "fund_cfo_to_ni") == late
    assert known(signals, "2014-Q4", "fund_roe_ttm") == late
    assert known(signals, "2015-Q1", "eps_ttm") < late
    assert known(signals, "2015-Q1", "fund_gm_yoy_chg") == late
    assert known(signals, "2015-Q1", "fund_opm_yoy_chg") == late
    assert known(signals, "2015-Q2", "fund_gm_yoy_chg") < late
    assert known(signals, "2015-Q4", "fund_eps_yoy") == late
    assert known(signals, "2015-Q4", "eps_ttm") < late
    assert known(signals, "2014-Q1", "fund_debt_ratio") == late
    assert known(signals, "2014-Q2", "fund_gross_margin") < late
