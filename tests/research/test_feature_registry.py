from collections import Counter

import pytest

from apps.research.features.registry import (
    FEATURES,
    Availability,
    Group,
    feature,
    feature_names,
)
from apps.research.features.transforms import STOCK_LEVEL
from apps.research.management.commands.ic_report import UNSCREENED

FIRST_BATCH_GROUPS = {
    Group.TECHNICAL: 22,
    Group.SIZE_LIQUIDITY: 3,
    Group.CHIP: 13,
    Group.FUNDAMENTAL: 16,
    Group.INDUSTRY_RISK: 4,
    Group.MARKET: 6,
}
SECOND_BATCH_GROUPS = {
    Group.TECHNICAL: 5,
    Group.CHIP: 4,
    Group.FUNDAMENTAL: 1,
    Group.INDUSTRY_RISK: 3,
    Group.MARKET: 1,
}
CONDITIONAL = {
    "size_log_mcap",
    "liq_dollar_vol_med_20d",
    "liq_amihud_20d",
    "tech_vol_20d",
    "tech_vol_60d",
    "risk_beta_250d",
    "risk_idio_vol_60d",
    "fund_days_since_fs",
    "ind_sector_id",
    "fund_days_to_ex_date",
    "mkt_days_to_quarter_end",
}
ANNOUNCE_DATE = {
    "fund_rev_yoy",
    "fund_rev_3m_yoy",
    "fund_rev_mom",
    "fund_ep",
    "fund_eps_yoy",
    "fund_roe_ttm",
    "fund_gross_margin",
    "fund_op_margin",
    "fund_debt_ratio",
    "fund_gm_yoy_chg",
    "fund_opm_yoy_chg",
    "fund_cfo_to_ni",
    "fund_rev_yoy_q",
    "fund_days_since_fs",
    "mkt_business_signal",
}
PUBLISH_DATE = {"chip_big_holder_pct_chg_1w", "chip_holder_count_chg_1w"}
US_CLOSE = {"mkt_sox_ret_5d", "mkt_tsm_adr_gap", "mkt_usdtwd_chg_20d"}


def test_first_batch_has_the_63_spec_features_and_quarterly_revenue_growth() -> None:
    first = [spec for spec in FEATURES if spec.batch == 1]

    assert len(first) == 64
    assert Counter(spec.group for spec in first) == FIRST_BATCH_GROUPS


def test_second_batch_has_the_14_spec_features_by_group() -> None:
    second = [spec for spec in FEATURES if spec.batch == 2]

    assert len(second) == 14
    assert Counter(spec.group for spec in second) == SECOND_BATCH_GROUPS


def test_names_are_unique_and_carry_their_group_prefix() -> None:
    prefixes = {
        Group.TECHNICAL: ("tech_",),
        Group.SIZE_LIQUIDITY: ("size_", "liq_"),
        Group.CHIP: ("chip_",),
        Group.FUNDAMENTAL: ("fund_",),
        Group.INDUSTRY_RISK: ("ind_", "risk_"),
        Group.MARKET: ("mkt_",),
    }

    assert len({spec.name for spec in FEATURES}) == len(FEATURES)
    assert all(spec.name.startswith(prefixes[spec.group]) for spec in FEATURES)


def test_conditional_features_follow_the_screening_rule() -> None:
    assert {spec.name for spec in FEATURES if spec.conditional} == CONDITIONAL


def test_availability_follows_the_spec_table() -> None:
    by_time = {
        Availability.ANNOUNCE_DATE: ANNOUNCE_DATE,
        Availability.PUBLISH_DATE: PUBLISH_DATE,
        Availability.US_CLOSE: US_CLOSE,
    }
    for availability, names in by_time.items():
        assert {spec.name for spec in FEATURES if spec.availability is availability} == names
    others = ANNOUNCE_DATE | PUBLISH_DATE | US_CLOSE
    assert all(
        spec.availability is Availability.TW_CLOSE for spec in FEATURES if spec.name not in others
    )


def test_only_market_features_are_market_level_and_only_the_sector_id_is_categorical() -> None:
    assert {spec.name for spec in FEATURES if spec.categorical} == {"ind_sector_id"}
    assert all(spec.market_level == (spec.group is Group.MARKET) for spec in FEATURES)


def test_lookups_by_group_and_batch() -> None:
    assert feature_names(Group.SIZE_LIQUIDITY) == [
        "size_log_mcap",
        "liq_dollar_vol_med_20d",
        "liq_amihud_20d",
    ]
    assert feature_names(Group.INDUSTRY_RISK, batch=2) == [
        "ind_sector_id",
        "ind_mom_neutral",
        "ind_ep_neutral",
    ]
    assert len(feature_names()) == 64
    assert feature("tech_limit_up_t").group is Group.TECHNICAL
    with pytest.raises(KeyError, match="tech_nope"):
        feature("tech_nope")


def test_the_second_batch_stays_out_of_screening_and_training() -> None:
    second = set(feature_names(batch=2))

    assert len(second) == 14
    assert not second & set(STOCK_LEVEL)
    assert not second & set(UNSCREENED)
