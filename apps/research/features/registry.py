"""Every SPEC §6 feature with its group, availability time, screening role and batch."""

from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatchcase


class Group(StrEnum):
    TECHNICAL = "technical"
    SIZE_LIQUIDITY = "size_liquidity"
    CHIP = "chip"
    FUNDAMENTAL = "fundamental"
    INDUSTRY_RISK = "industry_risk"
    MARKET = "market"


class Availability(StrEnum):
    TW_CLOSE = "tw_close"
    ANNOUNCE_DATE = "announce_date"
    PUBLISH_DATE = "publish_date"
    US_CLOSE = "us_close"
    LONDON_PRIOR_CLOSE = "london_prior_close"


CONDITIONAL_PATTERNS = (
    "size_*",
    "liq_*",
    "tech_vol_*",
    "risk_*",
    "fund_days_since_fs",
    "ind_sector_id",
    "*_days_to_*",
)
CATEGORICAL = frozenset({"ind_sector_id"})


@dataclass(frozen=True)
class FeatureSpec:
    """Conditional features skip the IC threshold of SPEC §6 處理步驟 7."""

    name: str
    group: Group
    availability: Availability
    batch: int

    @property
    def conditional(self) -> bool:
        return any(fnmatchcase(self.name, pattern) for pattern in CONDITIONAL_PATTERNS)

    @property
    def categorical(self) -> bool:
        return self.name in CATEGORICAL

    @property
    def market_level(self) -> bool:
        return self.group is Group.MARKET


def _specs(
    group: Group,
    names: str,
    availability: Availability = Availability.TW_CLOSE,
    batch: int = 1,
) -> list[FeatureSpec]:
    return [FeatureSpec(name, group, availability, batch) for name in names.split()]


TECH, SIZE, CHIP, FUND = Group.TECHNICAL, Group.SIZE_LIQUIDITY, Group.CHIP, Group.FUNDAMENTAL
RISK, MKT = Group.INDUSTRY_RISK, Group.MARKET
ANNOUNCE, PUBLISH, US = Availability.ANNOUNCE_DATE, Availability.PUBLISH_DATE, Availability.US_CLOSE
LONDON = Availability.LONDON_PRIOR_CLOSE

FEATURES: tuple[FeatureSpec, ...] = (
    *_specs(TECH, "tech_ret_1d tech_ret_5d tech_ret_20d tech_ret_60d tech_mom_250_20"),
    *_specs(TECH, "tech_ma_gap_5d tech_ma_gap_20d tech_ma_gap_60d tech_vol_20d tech_vol_60d"),
    *_specs(TECH, "tech_volume_ratio_5_20 tech_turnover_rate_5d"),
    *_specs(TECH, "tech_rsi_14 tech_kd_k tech_kd_d tech_macd_hist_pct"),
    *_specs(TECH, "tech_high_52w_gap tech_low_52w_gap"),
    *_specs(TECH, "tech_limit_up_t tech_limit_up_cnt_20d tech_limit_down_cnt_20d tech_attention_t"),
    *_specs(SIZE, "size_log_mcap liq_dollar_vol_med_20d liq_amihud_20d"),
    *_specs(CHIP, "chip_fi_net_1d chip_fi_net_5d chip_fi_net_20d"),
    *_specs(CHIP, "chip_it_net_5d chip_dealer_self_net_5d chip_dealer_hedge_net_5d"),
    *_specs(CHIP, "chip_margin_chg_5d chip_margin_usage chip_short_ratio"),
    *_specs(CHIP, "chip_fi_holding_pct chip_sbl_short_chg_5d"),
    *_specs(CHIP, "chip_big_holder_pct_chg_1w chip_holder_count_chg_1w", PUBLISH),
    *_specs(FUND, "fund_rev_yoy fund_rev_3m_yoy fund_rev_mom fund_ep fund_eps_yoy", ANNOUNCE),
    *_specs(FUND, "fund_bp fund_dividend_yield"),
    *_specs(FUND, "fund_roe_ttm fund_gross_margin fund_op_margin fund_debt_ratio", ANNOUNCE),
    *_specs(FUND, "fund_gm_yoy_chg fund_opm_yoy_chg fund_cfo_to_ni fund_days_since_fs", ANNOUNCE),
    *_specs(FUND, "fund_rev_yoy_q", ANNOUNCE),
    *_specs(RISK, "risk_beta_250d risk_idio_vol_60d ind_sector_ret_20d ind_rel_ret_20d"),
    *_specs(MKT, "mkt_taiex_ret_5d mkt_taiex_ret_20d"),
    *_specs(MKT, "mkt_sox_ret_5d mkt_tsm_adr_gap", US),
    *_specs(MKT, "mkt_usdtwd_chg_20d", LONDON),
    *_specs(MKT, "mkt_business_signal", ANNOUNCE),
    *_specs(TECH, "tech_parkinson_vol_20d tech_gap_1d tech_close_loc_5d", batch=2),
    *_specs(TECH, "tech_max_ret_20d tech_skew_60d", batch=2),
    *_specs(CHIP, "chip_day_trade_ratio_5d", batch=2),
    *_specs(CHIP, "chip_fi_holding_chg_20d chip_it_net_20d chip_fi_buy_streak", batch=2),
    *_specs(RISK, "ind_sector_id ind_mom_neutral ind_ep_neutral", batch=2),
    *_specs(FUND, "fund_days_to_ex_date", batch=2),
    *_specs(MKT, "mkt_days_to_quarter_end", batch=2),
)
BY_NAME = {spec.name: spec for spec in FEATURES}


def feature(name: str) -> FeatureSpec:
    """Registry entry of one feature; unknown names raise KeyError."""
    return BY_NAME[name]


def feature_names(group: Group | None = None, batch: int = 1) -> list[str]:
    """Names of one batch in spec order, optionally of one group."""
    return [
        spec.name
        for spec in FEATURES
        if spec.batch == batch and (group is None or spec.group is group)
    ]
