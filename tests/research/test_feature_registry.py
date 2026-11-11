from collections import Counter

import pytest

from apps.research.features.build import FEATURE_COLUMNS, FIRST_BATCH_COLUMNS, MODEL_COLUMNS
from apps.research.features.registry import (
    DORMANT,
    FEATURES,
    UNUSED,
    Availability,
    Group,
    feature,
    feature_names,
)
from apps.research.features.transforms import STOCK_LEVEL
from apps.research.management.commands.ic_report import UNSCREENED
from apps.research.production import DESIGNS
from apps.research.training import LINEAR_FEATURES, linear_features

FIRST_BATCH_GROUPS = {
    Group.TECHNICAL: 22,
    Group.SIZE_LIQUIDITY: 3,
    Group.CHIP: 13,
    Group.FUNDAMENTAL: 15,
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
US_CLOSE = {"mkt_sox_ret_5d", "mkt_tsm_adr_gap"}
LONDON_PRIOR_CLOSE = {"mkt_usdtwd_chg_20d"}


def test_first_batch_has_63_features_including_quarterly_revenue_growth() -> None:
    first = [spec for spec in FEATURES if spec.batch == 1]

    assert len(first) == 63
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
        Availability.LONDON_PRIOR_CLOSE: LONDON_PRIOR_CLOSE,
    }
    for availability, names in by_time.items():
        assert {spec.name for spec in FEATURES if spec.availability is availability} == names
    others = ANNOUNCE_DATE | PUBLISH_DATE | US_CLOSE | LONDON_PRIOR_CLOSE
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
    assert len(feature_names()) == 63
    assert feature("tech_limit_up_t").group is Group.TECHNICAL
    with pytest.raises(KeyError, match="tech_nope"):
        feature("tech_nope")


def test_the_second_batch_stays_out_of_screening_and_training() -> None:
    second = set(feature_names(batch=2))

    assert len(second) == 14
    assert not second & set(STOCK_LEVEL)
    assert not second & set(UNSCREENED)


def test_monthly_revenue_mom_is_stored_but_no_batch_reads_it() -> None:
    read = [*FIRST_BATCH_COLUMNS, *MODEL_COLUMNS, *STOCK_LEVEL, *UNSCREENED, *LINEAR_FEATURES]

    assert {spec.batch for spec in FEATURES} == {UNUSED, 1, 2}
    assert feature_names(batch=UNUSED) == ["fund_rev_mom"]
    assert [spec.name for spec in FEATURES] == FEATURE_COLUMNS
    assert [*feature_names(batch=1), *feature_names(batch=2)] == MODEL_COLUMNS
    assert "fund_rev_mom" in FEATURE_COLUMNS
    assert "fund_rev_mom" not in read
    with pytest.raises(ValueError, match="fund_rev_mom"):
        linear_features(["fund_ep", "fund_rev_mom"])


def test_dormant_features_keep_their_place_in_the_registry_and_the_frozen_design() -> None:
    assert DORMANT == ("chip_big_holder_pct_chg_1w", "chip_holder_count_chg_1w")
    assert set(DORMANT) <= set(STOCK_LEVEL)
    assert set(DORMANT) <= set(FEATURE_COLUMNS)
    assert set(DORMANT) <= set(FIRST_BATCH_COLUMNS)
    rank = DESIGNS["rank"].variant.features
    assert len(rank) == 60
    assert set(DORMANT) <= set(rank)
