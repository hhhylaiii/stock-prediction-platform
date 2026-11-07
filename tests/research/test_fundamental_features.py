import math
from datetime import date, timedelta

import pandas as pd
import pytest

from apps.research.features.fundamental import (
    fundamental_features,
    revenue_signals,
    revenue_sourcing,
    sourced_name,
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
            "revenue": 1000.0,
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


def test_revenue_sourcing_marks_signals_whose_months_are_all_on_record() -> None:
    revenue = months("2015-01", [0.0] + [100.0] * 19)
    holed = revenue[revenue["period"] != "2015-04"]

    full = revenue_sourcing(revenue)
    gaps = revenue_sourcing(holed)

    assert set(full["signal"]) == {sourced_name(name) for name in REVENUE_READS}
    assert one(full, "2016-01")[sourced_name("fund_rev_yoy")] == 1.0
    assert math.isnan(one(revenue_signals(revenue), "2016-01")["fund_rev_yoy"])
    assert one(full, "2016-03")[sourced_name("fund_rev_3m_yoy")] == 1.0
    assert one(gaps, "2016-04")[sourced_name("fund_rev_yoy")] == 0.0
    assert one(gaps, "2016-04")[sourced_name("fund_rev_mom")] == 1.0
    assert one(gaps, "2015-05")[sourced_name("fund_rev_mom")] == 0.0
    assert one(gaps, "2016-03")[sourced_name("fund_rev_yoy")] == 1.0
    assert one(gaps, "2016-03")[sourced_name("fund_rev_3m_yoy")] == 1.0
    assert one(gaps, "2016-06")[sourced_name("fund_rev_3m_yoy")] == 0.0


def test_revenue_sourcing_holds_months_masked_across_2013_as_on_record() -> None:
    revenue = months("2012-01", [100.0] * 12 + [400.0] * 15)

    sourcing = revenue_sourcing(revenue)

    assert math.isnan(one(revenue_signals(revenue), "2013-06")["fund_rev_yoy"])
    assert one(sourcing, "2013-06")[sourced_name("fund_rev_yoy")] == 1.0
    assert one(sourcing, "2012-06")[sourced_name("fund_rev_yoy")] == 0.0


REVENUE_READS = {
    "fund_rev_yoy": {0, 12},
    "fund_rev_3m_yoy": {0, 1, 2, 12, 13, 14},
    "fund_rev_mom": {0, 1},
}
STATEMENT_READS = {
    "eps_ttm": {0, 1, 2, 3},
    "fund_eps_yoy": set(range(8)),
    "fund_roe_ttm": {0, 1, 2, 3},
    "fund_gross_margin": {0},
    "fund_op_margin": {0},
    "fund_debt_ratio": {0},
    "fund_gm_yoy_chg": {0, 4},
    "fund_opm_yoy_chg": {0, 4},
    "fund_cfo_to_ni": {0, 1, 2, 3},
    "fund_rev_yoy_q": {0, 4},
}
LATE = pd.Timestamp("2030-01-02")


def later_by(signals: pd.DataFrame, first: str, k: int) -> str:
    periods = signals["period"].drop_duplicates().tolist()
    return str(periods[periods.index(first) + k])


@pytest.mark.parametrize("signal", list(REVENUE_READS))
def test_each_revenue_signal_waits_for_exactly_the_months_it_reads(signal: str) -> None:
    revenue = months("2015-01", [100.0] * 30)
    revenue.loc[revenue["period"] == "2015-03", "announce_date"] = LATE.date()

    signals = revenue_signals(revenue)

    waits = [known(signals, later_by(signals, "2015-03", k), signal) == LATE for k in range(15)]
    assert waits == [k in REVENUE_READS[signal] for k in range(15)]


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
    assert math.isnan(one(signals, "2016-Q1")["fund_cfo_to_ni"])


def test_quarterly_revenue_growth_compares_the_same_quarter_a_year_earlier() -> None:
    revenue = [100.0, 110.0, 120.0, 130.0, 150.0, 0.0, 132.0, 130.0, 160.0, 55.0]
    signals = statement_signals(quarters("2014-Q1", 10, revenue=revenue))

    growth = {period: one(signals, period)["fund_rev_yoy_q"] for period in ("2015-Q1", "2016-Q1")}
    assert growth == pytest.approx({"2015-Q1": 0.5, "2016-Q1": 160 / 150 - 1})
    assert one(signals, "2015-Q2")["fund_rev_yoy_q"] == pytest.approx(-1.0)
    assert one(signals, "2015-Q4")["fund_rev_yoy_q"] == pytest.approx(0.0)
    assert math.isnan(one(signals, "2014-Q4")["fund_rev_yoy_q"])
    assert math.isnan(one(signals, "2016-Q2")["fund_rev_yoy_q"])


SCOPE_READS = ["fund_cfo_to_ni", "fund_gm_yoy_chg", "fund_opm_yoy_chg", "fund_rev_yoy_q"]
PARENT_READS = ["eps_ttm", "fund_roe_ttm"]


def test_a_change_of_statement_basis_breaks_only_signals_of_consolidation_scope() -> None:
    switched = quarters("2012-Q1", 12).assign(fin_type=["individual"] * 4 + ["consolidated"] * 8)

    signals = statement_signals(switched)

    first = one(signals, "2013-Q1")
    assert first[SCOPE_READS].isna().all()
    assert first[PARENT_READS].tolist() == pytest.approx([4.0, 0.2])
    assert first["fund_gross_margin"] == pytest.approx(0.4)
    assert math.isnan(one(signals, "2013-Q3")["fund_cfo_to_ni"])
    fourth = one(signals, "2013-Q4")
    assert fourth["fund_cfo_to_ni"] == pytest.approx(1.2)
    assert fourth["fund_eps_yoy"] == pytest.approx(0.0)
    assert fourth[["fund_gm_yoy_chg", "fund_rev_yoy_q"]].isna().all()
    later = one(signals, "2014-Q1")
    assert later[["fund_gm_yoy_chg", "fund_rev_yoy_q"]].tolist() == pytest.approx([0.0, 0.0])


def test_a_quarter_filed_on_the_other_basis_breaks_the_quarters_around_it() -> None:
    kinds = ["consolidated"] * 4 + ["individual"] + ["consolidated"] * 4
    signals = statement_signals(quarters("2015-Q1", 9).assign(fin_type=kinds))

    assert [one(signals, f"2016-Q{q}")["eps_ttm"] for q in (1, 2, 4)] == pytest.approx([4.0] * 3)
    assert math.isnan(one(signals, "2016-Q2")["fund_cfo_to_ni"])
    assert math.isnan(one(signals, "2016-Q4")["fund_cfo_to_ni"])
    assert one(signals, "2017-Q1")["fund_cfo_to_ni"] == pytest.approx(1.2)


def test_monthly_signals_reading_both_sides_of_2013_01_are_missing() -> None:
    signals = revenue_signals(months("2012-01", [100.0] * 12 + [400.0] * 15))

    def value(period: str, name: str) -> float:
        return float(one(signals, period)[name])

    assert value("2012-12", "fund_rev_mom") == pytest.approx(0.0)
    assert math.isnan(value("2013-01", "fund_rev_mom"))
    assert value("2013-02", "fund_rev_mom") == pytest.approx(0.0)
    assert all(math.isnan(value(f"2013-{m:02d}", "fund_rev_yoy")) for m in range(1, 13))
    assert value("2014-01", "fund_rev_yoy") == pytest.approx(0.0)
    assert math.isnan(value("2014-02", "fund_rev_3m_yoy"))
    assert value("2014-03", "fund_rev_3m_yoy") == pytest.approx(0.0)


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
        fund_rev_yoy_q=0.05,
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


@pytest.mark.parametrize("signal", list(STATEMENT_READS))
def test_each_statement_signal_waits_for_exactly_the_quarters_it_reads(signal: str) -> None:
    statements = quarters("2013-Q1", 16)
    statements.loc[statements["period"] == "2013-Q2", "announce_date"] = LATE.date()

    signals = statement_signals(statements)

    waits = [known(signals, later_by(signals, "2013-Q2", k), signal) == LATE for k in range(8)]
    assert waits == [k in STATEMENT_READS[signal] for k in range(8)]


def test_the_first_announced_statement_of_a_quarter_is_kept() -> None:
    individual = quarters("2015-Q1", 2, gross_margin=[0.3, 0.3]).assign(fin_type="individual")
    consolidated = quarters("2015-Q1", 2, gross_margin=[0.5, 0.6])
    late = consolidated["period"] == "2015-Q1"
    consolidated.loc[late, "announce_date"] = date(2015, 9, 1)

    signals = statement_signals(pd.concat([consolidated, individual]))

    assert one(signals, "2015-Q1")["fund_gross_margin"] == pytest.approx(0.3)
    assert one(signals, "2015-Q2")["fund_gross_margin"] == pytest.approx(0.6)
