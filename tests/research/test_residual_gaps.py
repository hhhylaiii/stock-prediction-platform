import json
import math
from datetime import date

import pandas as pd
import pytest

from apps.research.features.build import MODEL_COLUMNS
from apps.research.features.fundamental import STATEMENT_INPUTS, sourced_name
from apps.research.features.residual import (
    ACCEPT_GAP,
    ACCEPT_RATIO,
    FALLBACKS,
    MASKED_MONTHS,
    MAX_MASKED_SHARE,
    OBSERVABLE,
    affected_months,
    apply_fallbacks,
    mask_months,
    monthly_coverage,
    observable_gaps,
    observable_name,
    rules_signature,
)

NAN = math.nan
COLUMNS = ["fund_rev_yoy", "fund_rev_3m_yoy", "fund_rev_mom", "fund_rev_yoy_q"]


def d(text: str) -> date:
    return date.fromisoformat(text)


def frame(rows: list[tuple[str, str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": [row[0] for row in rows],
            "date": [d(row[1]) for row in rows],
            **{name: [row[k + 2] for row in rows] for k, name in enumerate(COLUMNS)},
        }
    )


def sourcing(flags: list[float]) -> pd.DataFrame:
    """The same on-record flag for every revenue signal of each row."""
    names = ["fund_rev_yoy", "fund_rev_3m_yoy", "fund_rev_mom"]
    return pd.DataFrame({sourced_name(name): flags for name in names})


def test_revenue_growth_takes_the_quarterly_growth_only_where_its_months_are_off_record() -> None:
    features = frame(
        [
            ("2330", "2017-03-01", 0.10, 0.20, 0.01, 0.30),
            ("2330", "2017-03-02", NAN, NAN, NAN, 0.30),
            ("2317", "2017-03-02", NAN, NAN, NAN, 0.40),
            ("1101", "2017-03-02", NAN, NAN, NAN, 0.50),
            ("2002", "2017-03-02", NAN, NAN, NAN, NAN),
        ]
    )

    ruled = apply_fallbacks(features, sourcing([1.0, NAN, 0.0, 1.0, NAN]))

    assert ruled["fund_rev_yoy"].tolist()[:3] == [0.10, 0.30, 0.40]
    assert ruled["fund_rev_3m_yoy"].tolist()[:3] == [0.20, 0.30, 0.40]
    assert math.isnan(ruled["fund_rev_yoy"].iloc[3])
    assert math.isnan(ruled["fund_rev_yoy"].iloc[4])
    assert ruled["fund_rev_mom"].isna().sum() == 4
    assert ruled["fund_rev_yoy_q"].tolist()[:4] == [0.30, 0.30, 0.40, 0.50]
    assert features["fund_rev_yoy"].isna().sum() == 4


def test_each_fallback_reads_its_own_on_record_flag() -> None:
    features = frame([("2330", "2017-03-02", NAN, NAN, NAN, 0.30)])
    flags = sourcing([1.0]).assign(**{sourced_name("fund_rev_3m_yoy"): [0.0]})

    ruled = apply_fallbacks(features, flags)

    assert math.isnan(ruled["fund_rev_yoy"].iloc[0])
    assert ruled["fund_rev_3m_yoy"].iloc[0] == 0.30


def test_any_column_is_masked_for_every_stock_in_its_affected_months() -> None:
    features = frame(
        [
            ("2330", "2017-02-28", 0.1, 0.1, 0.01, 0.1),
            ("2330", "2017-03-01", 0.1, 0.1, 0.02, 0.1),
            ("1101", "2017-03-31", 0.1, 0.1, 0.03, 0.1),
            ("1101", "2017-04-03", 0.1, 0.1, 0.04, 0.1),
        ]
    ).assign(ind_sector_id=[1.0, 2.0, 3.0, 4.0])
    rules = {"fund_rev_mom": frozenset({"2017-03"}), "ind_sector_id": frozenset({"2017-04"})}

    masked = mask_months(features, rules)

    assert masked["fund_rev_mom"].tolist()[0] == 0.01
    assert masked["fund_rev_mom"].iloc[1:3].isna().all()
    assert masked["fund_rev_mom"].tolist()[3] == 0.04
    assert masked["ind_sector_id"].tolist()[:3] == [1.0, 2.0, 3.0]
    assert math.isnan(masked["ind_sector_id"].iloc[3])
    others = ["fund_rev_mom", "ind_sector_id"]
    pd.testing.assert_frame_equal(masked.drop(columns=others), features.drop(columns=others))
    assert features["fund_rev_mom"].notna().all()


def test_defaults_are_the_decided_rules_and_empty_rules_change_nothing() -> None:
    assert dict(FALLBACKS) == {
        "fund_rev_yoy": "fund_rev_yoy_q",
        "fund_rev_3m_yoy": "fund_rev_yoy_q",
    }
    assert dict(MASKED_MONTHS) == {}
    assert (ACCEPT_RATIO, ACCEPT_GAP, MAX_MASKED_SHARE) == (1.5, 0.05, 0.2)
    features = frame([("2330", "2017-03-02", NAN, 0.2, 0.01, 0.30)])

    pd.testing.assert_frame_equal(apply_fallbacks(features, sourcing([NAN]), {}), features)
    pd.testing.assert_frame_equal(mask_months(features), features)
    pd.testing.assert_frame_equal(mask_months(features, {}), features)


def test_monthly_coverage_splits_universe_rows_by_later_delisting() -> None:
    rows = pd.DataFrame(
        {
            "stock_id": ["2330", "2317", "1101", "9999", "2330", "1101"],
            "date": [d(text) for text in ["2017-03-01"] * 4 + ["2017-04-03"] * 2],
            "in_universe": [True, True, True, False, True, True],
            "fund_rev_mom": [0.1, NAN, NAN, 0.3, 0.1, 0.2],
        }
    )
    delisted = pd.Series({"2330": False, "2317": False, "1101": True, "9999": True})

    coverage = monthly_coverage(rows, "fund_rev_mom", delisted)

    assert coverage.loc["2017-03", "current"] == 0.5
    assert coverage.loc["2017-03", "delisted"] == 0.0
    assert coverage.loc["2017-04", "delisted"] == 1.0
    no_delisted = monthly_coverage(rows[rows["stock_id"] != "1101"], "fund_rev_mom", delisted)
    assert no_delisted["delisted"].isna()["2017-03"]


@pytest.mark.parametrize(
    ("current", "delisted", "affected"),
    [
        (0.99, 0.90, True),
        (0.99, 0.94, False),
        (0.93, 0.88, False),
        (0.80, 0.70, False),
        (0.80, 0.69, True),
        (1.00, 0.95, False),
        (1.00, 0.94, True),
        (0.99, NAN, False),
        (0.00, 0.10, True),
        (0.02, 0.10, True),
        (0.00, 0.05, False),
        (0.90, 0.99, False),
    ],
)
def test_a_month_is_affected_when_it_fails_the_acceptance_on_its_own(
    current: float, delisted: float, affected: bool
) -> None:
    coverage = pd.DataFrame({"current": [current], "delisted": [delisted]}, index=["2017-03"])

    assert affected_months(coverage) == (["2017-03"] if affected else [])


def test_affected_months_are_sorted() -> None:
    coverage = pd.DataFrame(
        {"current": [1.0, 1.0, 1.0], "delisted": [0.5, 1.0, 0.5]},
        index=["2017-05", "2017-04", "2017-03"],
    )

    assert affected_months(coverage) == ["2017-03", "2017-05"]


def test_the_rules_signature_names_the_fallbacks_and_the_masked_months() -> None:
    signature = json.loads(
        rules_signature(
            {"fund_rev_yoy": "fund_rev_yoy_q"}, {"fund_rev_mom": frozenset({"2017-04", "2017-03"})}
        )
    )

    assert signature == {
        "fallbacks": {"fund_rev_yoy": "fund_rev_yoy_q"},
        "masked": {"fund_rev_mom": ["2017-03", "2017-04"]},
        "fills": "off_record",
    }
    assert rules_signature() == rules_signature(FALLBACKS, MASKED_MONTHS)
    assert rules_signature() != rules_signature(FALLBACKS, {"fund_rev_mom": frozenset({"2017-03"})})


MARGINS = ["fund_gross_margin", "fund_op_margin", "fund_gm_yoy_chg", "fund_opm_yoy_chg"]


def test_a_gap_is_observable_only_when_every_input_its_value_reads_is_on_record() -> None:
    panel = pd.DataFrame(
        {
            "stock_id": ["2330", "2881", "1101", "1102"],
            "date": [d("2017-03-01")] * 4,
            "industry": ["半導體業", "金融業", "水泥工業", None],
            **{sourced_name(name): [1.0, 0.0, NAN, 0.0] for name in STATEMENT_INPUTS},
            sourced_name("fund_rev_yoy"): [1.0, 0.0, NAN, 0.0],
            sourced_name("fund_rev_3m_yoy"): [0.0, 0.0, 0.0, 1.0],
        }
    )
    panel.loc[[0, 3], sourced_name("fund_rev_yoy_q")] = 1.0

    gaps = observable_gaps(panel)

    def column(name: str) -> list[bool]:
        return gaps[observable_name(name)].tolist()

    assert list(gaps.columns) == ["stock_id", "date", *map(observable_name, OBSERVABLE)]
    assert set(OBSERVABLE) <= set(MODEL_COLUMNS)
    assert set(OBSERVABLE) == {
        *(name for name in STATEMENT_INPUTS if name != "eps_ttm"),
        "fund_ep",
        "ind_ep_neutral",
        "fund_rev_yoy",
        "fund_rev_3m_yoy",
    }
    for name in ("fund_ep", "ind_ep_neutral", "fund_eps_yoy", "fund_debt_ratio", "fund_cfo_to_ni"):
        assert column(name) == [True, False, False, False]
    for name in MARGINS:
        assert column(name) == [True, True, False, False]
    assert column("fund_rev_yoy") == [True, False, False, True]
    assert column("fund_rev_3m_yoy") == [True, False, False, True]


def test_monthly_coverage_counts_an_observable_gap_as_covered() -> None:
    rows = pd.DataFrame(
        {
            "stock_id": ["2330", "1101", "2330", "1101"],
            "date": [d("2017-03-01")] * 2 + [d("2017-03-02")] * 2,
            "in_universe": True,
            "fund_eps_yoy": [0.1, NAN, 0.1, NAN],
            observable_name("fund_eps_yoy"): [False, True, False, None],
        }
    )
    delisted = pd.Series({"2330": False, "1101": True})

    observed = monthly_coverage(rows, "fund_eps_yoy", delisted)
    plain = monthly_coverage(
        rows.drop(columns=observable_name("fund_eps_yoy")), "fund_eps_yoy", delisted
    )

    assert observed.loc["2017-03", "delisted"] == 0.5
    assert plain.loc["2017-03", "delisted"] == 0.0
    assert observed.loc["2017-03", "current"] == 1.0
