import hashlib
import json
import math
from datetime import date

import pandas as pd
import pytest

from apps.research.features.build import MODEL_COLUMNS
from apps.research.features.chip import SOURCED_CHIPS
from apps.research.features.fundamental import (
    BASIS_SIGNALS,
    STATEMENT_INPUTS,
    crossed_name,
    sourced_name,
)
from apps.research.features.residual import (
    ACCEPT_GAP,
    ACCEPT_RATIO,
    FALLBACKS,
    LISTING_WINDOWS,
    MASKED_MONTHS,
    MAX_FILL_RATIO,
    MAX_MASKED_SHARE,
    OBSERVABLE,
    affected_months,
    apply_fallbacks,
    basis_counts,
    fallback_filled,
    fill_flags,
    mask_months,
    masked_as_observable,
    monthly_coverage,
    observable_gaps,
    observable_name,
    rules_signature,
    unmasked_fills,
)
from apps.research.features.technical import MOMENTUM_DAYS

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
    assert (ACCEPT_RATIO, ACCEPT_GAP, MAX_MASKED_SHARE) == (1.5, 0.05, 0.2)
    features = frame([("2330", "2017-03-02", NAN, 0.2, 0.01, 0.30)])

    pd.testing.assert_frame_equal(apply_fallbacks(features, sourcing([NAN]), {}), features)
    pd.testing.assert_frame_equal(mask_months(features, {}), features)


def research_months() -> list[str]:
    return [f"{year}-{month:02d}" for year in range(2010, 2025) for month in range(1, 13)]


def test_masked_months_are_the_months_residual_coverage_found_on_2026_11_08() -> None:
    months = research_months()
    leading = frozenset(months[: months.index("2013-05") + 1])

    assert set(MASKED_MONTHS) == {
        "fund_rev_yoy",
        "fund_rev_3m_yoy",
        "fund_ep",
        "fund_eps_yoy",
        "fund_roe_ttm",
        "fund_gross_margin",
        "fund_op_margin",
        "fund_debt_ratio",
        "fund_gm_yoy_chg",
        "fund_opm_yoy_chg",
        "fund_cfo_to_ni",
        "fund_days_since_fs",
        "fund_rev_yoy_q",
        "ind_ep_neutral",
        "fund_days_to_ex_date",
    }
    for name in ("fund_gross_margin", "fund_op_margin", "fund_days_since_fs"):
        assert MASKED_MONTHS[name] == leading
    assert MASKED_MONTHS["fund_days_to_ex_date"] == frozenset({"2022-11"})
    assert MASKED_MONTHS["fund_ep"] == MASKED_MONTHS["ind_ep_neutral"]
    assert MASKED_MONTHS["fund_rev_yoy"] == frozenset(months[: months.index("2015-02") + 1])
    assert MASKED_MONTHS["fund_rev_3m_yoy"] == frozenset(
        [*months[: months.index("2015-04") + 1], "2016-03", "2017-01"]
    )
    assert len(MASKED_MONTHS["fund_rev_yoy"]) == 62
    assert len(MASKED_MONTHS["fund_eps_yoy"]) == 87
    assert sum(map(len, MASKED_MONTHS.values())) == 717
    text = json.dumps({name: sorted(months) for name, months in sorted(MASKED_MONTHS.items())})
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "f291a4bfc639b13d506d59f65405791db51289f3e95e70d95d6b92283b0b86b7"
    )
    for name, chosen in MASKED_MONTHS.items():
        assert chosen <= set(months), name
        assert len(chosen) <= len(months) / 2, name
    assert set(MASKED_MONTHS) <= set(MODEL_COLUMNS)


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
    shares = {"current": [current], "delisted": [delisted]}
    coverage = pd.DataFrame(
        {**shares, **{f"present_{k}": v for k, v in shares.items()}}, ["2017-03"]
    )

    assert affected_months(coverage) == (["2017-03"] if affected else [])


def test_affected_months_are_sorted() -> None:
    coverage = pd.DataFrame(
        {
            "current": [1.0, 1.0, 1.0],
            "delisted": [0.5, 1.0, 0.5],
            "present_current": [1.0, 1.0, 1.0],
            "present_delisted": [0.5, 1.0, 0.5],
        },
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
            "age": MOMENTUM_DAYS,
            **{sourced_name(name): [1.0, 0.0, NAN, 0.0] for name in STATEMENT_INPUTS},
            **{
                sourced_name(name): [float(k == row) for k in range(4)]
                for row, name in enumerate(SOURCED_CHIPS)
            },
            sourced_name("fund_rev_yoy"): [1.0, 1.0, NAN, 0.0],
            sourced_name("fund_rev_3m_yoy"): [0.0, 0.0, 0.0, 1.0],
        }
    )
    panel.loc[[0, 3], sourced_name("fund_rev_yoy_q")] = 1.0
    panel = panel.assign(**{crossed_name(name): [0.0, 1.0, NAN, 0.0] for name in BASIS_SIGNALS})
    panel.loc[3, crossed_name("fund_rev_yoy_q")] = 1.0

    gaps = observable_gaps(panel)

    def column(name: str) -> list[bool]:
        return gaps[observable_name(name)].tolist()

    assert list(gaps.columns) == [
        "stock_id",
        "date",
        *map(observable_name, OBSERVABLE),
        *map(observable_name, LISTING_WINDOWS),
        *map(crossed_name, BASIS_SIGNALS),
    ]
    assert gaps[crossed_name("fund_cfo_to_ni")].tolist() == [False, True, False, False]
    assert set(OBSERVABLE) <= set(MODEL_COLUMNS)
    assert not set(OBSERVABLE) & set(LISTING_WINDOWS)
    assert set(OBSERVABLE) == {
        *(name for name in STATEMENT_INPUTS if name != "eps_ttm"),
        "fund_ep",
        "ind_ep_neutral",
        "fund_rev_yoy",
        "fund_rev_3m_yoy",
        *SOURCED_CHIPS,
    }
    for row, name in enumerate(SOURCED_CHIPS):
        assert column(name) == [k == row for k in range(4)]
    paired = {*MARGINS, "fund_rev_yoy", "fund_rev_3m_yoy", "fund_rev_yoy_q", *SOURCED_CHIPS}
    for name in (name for name in OBSERVABLE if name not in paired):
        assert column(name) == [True, False, False, False]
    for name in MARGINS:
        assert column(name) == [True, True, False, False]
    assert column("fund_rev_yoy") == [True, True, False, False]
    assert column("fund_rev_3m_yoy") == [True, False, False, True]
    assert column("fund_rev_yoy_q") == [True, False, False, True]


def test_momentum_gaps_of_a_listing_younger_than_its_window_have_a_known_cause() -> None:
    signals = {signal for names in OBSERVABLE.values() for signal in names}
    ages = [0, MOMENTUM_DAYS - 1, MOMENTUM_DAYS, MOMENTUM_DAYS + 1]
    panel = pd.DataFrame(
        {
            "stock_id": ["6488"] * 4,
            "date": [d("2017-03-01"), d("2017-03-02"), d("2017-03-03"), d("2017-03-06")],
            "industry": "半導體業",
            "age": ages,
            **{sourced_name(name): 1.0 for name in signals},
            **{crossed_name(name): 0.0 for name in BASIS_SIGNALS},
        }
    )

    gaps = observable_gaps(panel)

    assert LISTING_WINDOWS == {"tech_mom_250_20": MOMENTUM_DAYS, "ind_mom_neutral": MOMENTUM_DAYS}
    assert set(LISTING_WINDOWS) <= set(MODEL_COLUMNS)
    for name in LISTING_WINDOWS:
        assert gaps[observable_name(name)].tolist() == [True, True, False, False]


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


def test_masked_months_count_as_gaps_with_a_known_cause() -> None:
    rows = pd.DataFrame(
        {
            "stock_id": ["2330", "1101", "2330"],
            "date": [d("2017-03-01"), d("2017-03-01"), d("2017-04-03")],
            "fund_eps_yoy": [NAN, NAN, NAN],
            "chip_short_ratio": [NAN, NAN, NAN],
            observable_name("fund_eps_yoy"): [True, None, False],
        }
    )
    masked = {"fund_eps_yoy": frozenset({"2017-03"}), "chip_short_ratio": frozenset({"2017-04"})}

    marked = masked_as_observable(rows, masked)

    assert marked[observable_name("fund_eps_yoy")].tolist() == [True, True, False]
    assert marked[observable_name("chip_short_ratio")].tolist() == [False, False, True]
    assert rows[observable_name("fund_eps_yoy")].tolist() == [True, None, False]
    pd.testing.assert_frame_equal(masked_as_observable(rows, {}), rows)


def test_basis_counts_take_gaps_read_across_report_bases_by_later_fate() -> None:
    rows = pd.DataFrame(
        {
            "stock_id": ["2330", "2317", "1101", "1101", "9999"],
            "date": [d("2017-03-01")] * 5,
            "in_universe": [True, True, True, True, False],
            "fund_rev_yoy_q": [NAN, NAN, NAN, 0.1, NAN],
            crossed_name("fund_rev_yoy_q"): [True, None, True, False, True],
        }
    )
    delisted = pd.Series({"2330": False, "2317": False, "1101": True, "9999": True})

    counts = basis_counts(rows, "fund_rev_yoy_q", delisted)

    assert counts.to_dict() == {"current": 1, "delisted": 1, "rows_current": 2, "rows_delisted": 2}


def test_a_month_where_only_the_delisted_have_values_stays_affected() -> None:
    days = [d("2013-03-01")] * 200
    rows = pd.DataFrame(
        {
            "stock_id": [f"c{k}" for k in range(100)] + [f"g{k}" for k in range(100)],
            "date": days,
            "in_universe": True,
            "fund_rev_yoy": [NAN] * 100 + [0.2] * 29 + [NAN] * 71,
            "fund_rev_yoy_q": NAN,
            observable_name("fund_rev_yoy"): [True] * 100 + [False] * 29 + [True] * 71,
        }
    )
    delisted = pd.Series(
        {**{f"c{k}": False for k in range(100)}, **{f"g{k}": True for k in range(100)}}
    )

    coverage = monthly_coverage(rows, "fund_rev_yoy", delisted)

    assert list(coverage.columns) == [
        "delisted",
        "current",
        "present_delisted",
        "present_current",
        "filled_delisted",
        "filled_current",
        "rows_delisted",
        "rows_current",
    ]
    assert list(coverage.loc["2013-03"]) == [1.0, 1.0, 0.29, 0.0, 0.0, 0.0, 100, 100]
    assert affected_months(coverage) == ["2013-03"]


def test_monthly_coverage_takes_fill_shares_and_rows_for_a_fallback_column() -> None:
    rows = pd.DataFrame(
        {
            "stock_id": ["2330", "2317", "1101", "9999", "2330"],
            "date": [d("2017-03-01")] * 4 + [d("2017-04-03")],
            "in_universe": [True, True, True, False, True],
            "fund_rev_yoy": [0.1, 0.2, 0.4, 0.5, NAN],
            "fund_rev_yoy_q": [0.1, 0.3, 0.4, 0.5, NAN],
        }
    )
    delisted = pd.Series({"2330": False, "2317": False, "1101": True, "9999": True})

    coverage = monthly_coverage(rows, "fund_rev_yoy", delisted)
    plain = monthly_coverage(rows, "fund_rev_yoy_q", delisted)

    assert fallback_filled(rows, "fund_rev_yoy").tolist() == [True, False, True, True, False]
    march, april = coverage.loc["2017-03"], coverage.loc["2017-04"]
    assert (march["filled_current"], march["filled_delisted"]) == (0.5, 1.0)
    assert (march["rows_current"], march["rows_delisted"]) == (2, 1)
    assert (april["rows_current"], april["rows_delisted"]) == (1, 0)
    assert coverage["filled_delisted"].isna().tolist() == [False, True]
    assert "filled_current" not in plain
    assert plain.loc["2017-03", "rows_current"] == 2


@pytest.mark.parametrize(
    ("current", "delisted", "affected"),
    [
        (0.00, 0.10, True),
        (0.10, 0.16, True),
        (0.10, 0.15, False),
        (0.20, 0.30, False),
        (0.20, 0.31, True),
        (0.02, 0.06, False),
        (0.00, 0.03, True),
        (0.00, 0.00, False),
        (0.05, NAN, False),
        (NAN, 0.03, False),
    ],
)
def test_a_month_is_affected_when_its_fills_lean_on_the_delisted(
    current: float, delisted: float, affected: bool
) -> None:
    coverage = pd.DataFrame(
        {
            "current": [1.0],
            "delisted": [1.0],
            "present_current": [1.0],
            "present_delisted": [1.0],
            "filled_current": [current],
            "filled_delisted": [delisted],
        },
        ["2017-03"],
    )

    assert affected_months(coverage) == (["2017-03"] if affected else [])


def test_unmasked_fills_weigh_the_months_left_by_their_rows() -> None:
    coverage = pd.DataFrame(
        {
            "filled_current": [0.1, 0.4, 0.9],
            "filled_delisted": [0.5, NAN, 0.9],
            "rows_current": [30, 10, 50],
            "rows_delisted": [4, 0, 6],
        },
        ["2017-03", "2017-04", "2017-05"],
    )

    shares = unmasked_fills(coverage, ["2017-05"])
    gone = unmasked_fills(coverage, coverage.index)

    assert shares["current"] == pytest.approx((0.1 * 30 + 0.4 * 10) / 40)
    assert shares["delisted"] == pytest.approx(0.5)
    assert gone.isna().tolist() == [True, True]


def test_fill_flags_mark_fills_over_the_ratio_or_on_the_delisted_only() -> None:
    current = pd.Series([0.01, 0.01, 1 / 78, 0.0, 0.0, NAN])
    delisted = pd.Series([0.0301, 0.03, 1 / 26, 0.01, 0.0, 0.1])

    assert MAX_FILL_RATIO == 3.0
    assert fill_flags(current, delisted).tolist() == [True, False, False, True, False, False]
