"""Research-layer tables."""

from django.db import models
from django.utils import timezone

from apps.market.models import finite
from apps.research.features.registry import feature_names


class HoldoutScope(models.TextChoices):
    RANK = "rank", "排名"
    RETURN = "return", "絕對報酬"
    INTERVAL = "interval", "區間與機率"


class HoldoutRun(models.Model):
    """One claimed holdout run per scope and attempt; attempts after the first carry a reason."""

    scope = models.CharField(max_length=16, choices=HoldoutScope.choices)
    attempt = models.PositiveSmallIntegerField()
    run_key = models.CharField(max_length=64)
    last_date = models.DateField()
    reason = models.TextField(blank=True, default="")
    claimed_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "holdout_run"
        constraints = [
            models.UniqueConstraint(fields=["scope", "attempt"], name="holdout_run_scope_attempt"),
            models.UniqueConstraint(fields=["run_key"], name="holdout_run_run_key"),
            models.CheckConstraint(
                condition=models.Q(scope__in=HoldoutScope.values), name="holdout_run_scope_known"
            ),
            models.CheckConstraint(
                condition=models.Q(attempt__gte=1), name="holdout_run_attempt_gte_1"
            ),
            models.CheckConstraint(
                condition=models.Q(attempt=1) | models.Q(reason__regex=r"\S"),
                name="holdout_run_rerun_has_reason",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.scope} #{self.attempt} {self.run_key}"


class ModelObjective(models.TextChoices):
    RANK = "rank", "排名"
    REGRESSION = "regression", "絕對報酬"
    QUANTILE = "quantile", "分位數"
    VOLATILITY = "volatility", "波動率"
    CLASSIFICATION = "classification", "上漲機率"


class ModelStatus(models.TextChoices):
    CANDIDATE = "candidate", "候選"
    ACTIVE = "active", "上線"
    DEGRADED = "degraded", "降級"
    RETIRED = "retired", "退役"


class ModelRun(models.Model):
    """One trained model: its window, parameters, features, versions and fold metrics (SPEC §7)."""

    run_id = models.CharField(max_length=64, primary_key=True)
    run_key = models.CharField(max_length=64, db_index=True)
    model_type = models.CharField(max_length=32)
    objective = models.CharField(max_length=16, choices=ModelObjective.choices)
    version = models.CharField(max_length=32)
    train_start = models.DateField()
    train_end = models.DateField()
    params = models.JSONField(default=dict, blank=True)
    feature_list = models.JSONField(default=list, blank=True)
    seed = models.IntegerField(null=True, blank=True)
    lib_versions = models.JSONField(default=dict)
    feature_hash = models.CharField(max_length=64)
    status = models.CharField(
        max_length=16, choices=ModelStatus.choices, default=ModelStatus.CANDIDATE
    )
    ic_mean = models.FloatField(null=True, blank=True)
    icir = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "model_run"
        constraints = [
            models.CheckConstraint(
                condition=models.Q(objective__in=ModelObjective.values),
                name="model_run_objective_known",
            ),
            models.CheckConstraint(
                condition=models.Q(status__in=ModelStatus.values), name="model_run_status_known"
            ),
            models.CheckConstraint(
                condition=models.Q(train_start__lte=models.F("train_end")),
                name="model_run_window_ordered",
            ),
            models.CheckConstraint(condition=finite("ic_mean", "icir"), name="model_run_finite"),
        ]

    def __str__(self) -> str:
        return f"{self.run_id} {self.status}"


class LabelDaily(models.Model):
    """Forward labels per stock, day and horizon; vol and up_5pct sit on the 20-day rows."""

    pk = models.CompositePrimaryKey("stock_id", "date", "horizon")
    stock_id = models.CharField(max_length=10)
    date = models.DateField()
    horizon = models.PositiveSmallIntegerField()
    fwd_ret = models.FloatField(null=True, blank=True)
    fwd_ret_open = models.FloatField(null=True, blank=True)
    fwd_rank_pct = models.FloatField(null=True, blank=True)
    fwd_vol_20d = models.FloatField(null=True, blank=True)
    fwd_up_5pct = models.BooleanField(null=True, blank=True)
    untradable_t1 = models.BooleanField(null=True, blank=True)

    class Meta:
        db_table = "label_daily"
        indexes = [models.Index(fields=["date", "horizon"], name="label_daily_date_horizon_idx")]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(horizon__in=[5, 20]), name="label_daily_horizon_known"
            ),
            models.CheckConstraint(
                condition=finite("fwd_ret", "fwd_ret_open", "fwd_vol_20d"),
                name="label_daily_finite",
            ),
            models.CheckConstraint(
                condition=models.Q(fwd_rank_pct__gt=0, fwd_rank_pct__lte=1)
                | models.Q(fwd_rank_pct__isnull=True),
                name="label_daily_rank_pct_range",
            ),
            models.CheckConstraint(
                condition=models.Q(fwd_vol_20d__gte=0) | models.Q(fwd_vol_20d__isnull=True),
                name="label_daily_vol_gte_0",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.date} {self.horizon}d"


class FeatureDaily(models.Model):
    """Raw first-batch features per stock and trading day; transforms run in the pipeline."""

    pk = models.CompositePrimaryKey("stock_id", "date")
    stock_id = models.CharField(max_length=10)
    date = models.DateField()
    in_universe = models.BooleanField()
    tech_ret_1d = models.FloatField(null=True, blank=True)
    tech_ret_5d = models.FloatField(null=True, blank=True)
    tech_ret_20d = models.FloatField(null=True, blank=True)
    tech_ret_60d = models.FloatField(null=True, blank=True)
    tech_mom_250_20 = models.FloatField(null=True, blank=True)
    tech_ma_gap_5d = models.FloatField(null=True, blank=True)
    tech_ma_gap_20d = models.FloatField(null=True, blank=True)
    tech_ma_gap_60d = models.FloatField(null=True, blank=True)
    tech_vol_20d = models.FloatField(null=True, blank=True)
    tech_vol_60d = models.FloatField(null=True, blank=True)
    tech_volume_ratio_5_20 = models.FloatField(null=True, blank=True)
    tech_turnover_rate_5d = models.FloatField(null=True, blank=True)
    tech_rsi_14 = models.FloatField(null=True, blank=True)
    tech_kd_k = models.FloatField(null=True, blank=True)
    tech_kd_d = models.FloatField(null=True, blank=True)
    tech_macd_hist_pct = models.FloatField(null=True, blank=True)
    tech_high_52w_gap = models.FloatField(null=True, blank=True)
    tech_low_52w_gap = models.FloatField(null=True, blank=True)
    tech_limit_up_t = models.FloatField(null=True, blank=True)
    tech_limit_up_cnt_20d = models.FloatField(null=True, blank=True)
    tech_limit_down_cnt_20d = models.FloatField(null=True, blank=True)
    tech_attention_t = models.FloatField(null=True, blank=True)
    size_log_mcap = models.FloatField(null=True, blank=True)
    liq_dollar_vol_med_20d = models.FloatField(null=True, blank=True)
    liq_amihud_20d = models.FloatField(null=True, blank=True)
    chip_fi_net_1d = models.FloatField(null=True, blank=True)
    chip_fi_net_5d = models.FloatField(null=True, blank=True)
    chip_fi_net_20d = models.FloatField(null=True, blank=True)
    chip_it_net_5d = models.FloatField(null=True, blank=True)
    chip_dealer_self_net_5d = models.FloatField(null=True, blank=True)
    chip_dealer_hedge_net_5d = models.FloatField(null=True, blank=True)
    chip_margin_chg_5d = models.FloatField(null=True, blank=True)
    chip_margin_usage = models.FloatField(null=True, blank=True)
    chip_short_ratio = models.FloatField(null=True, blank=True)
    chip_fi_holding_pct = models.FloatField(null=True, blank=True)
    chip_sbl_short_chg_5d = models.FloatField(null=True, blank=True)
    chip_big_holder_pct_chg_1w = models.FloatField(null=True, blank=True)
    chip_holder_count_chg_1w = models.FloatField(null=True, blank=True)
    fund_rev_yoy = models.FloatField(null=True, blank=True)
    fund_rev_3m_yoy = models.FloatField(null=True, blank=True)
    fund_rev_mom = models.FloatField(null=True, blank=True)
    fund_ep = models.FloatField(null=True, blank=True)
    fund_eps_yoy = models.FloatField(null=True, blank=True)
    fund_bp = models.FloatField(null=True, blank=True)
    fund_dividend_yield = models.FloatField(null=True, blank=True)
    fund_roe_ttm = models.FloatField(null=True, blank=True)
    fund_gross_margin = models.FloatField(null=True, blank=True)
    fund_op_margin = models.FloatField(null=True, blank=True)
    fund_debt_ratio = models.FloatField(null=True, blank=True)
    fund_gm_yoy_chg = models.FloatField(null=True, blank=True)
    fund_opm_yoy_chg = models.FloatField(null=True, blank=True)
    fund_cfo_to_ni = models.FloatField(null=True, blank=True)
    fund_days_since_fs = models.FloatField(null=True, blank=True)
    risk_beta_250d = models.FloatField(null=True, blank=True)
    risk_idio_vol_60d = models.FloatField(null=True, blank=True)
    ind_sector_ret_20d = models.FloatField(null=True, blank=True)
    ind_rel_ret_20d = models.FloatField(null=True, blank=True)
    mkt_taiex_ret_5d = models.FloatField(null=True, blank=True)
    mkt_taiex_ret_20d = models.FloatField(null=True, blank=True)
    mkt_sox_ret_5d = models.FloatField(null=True, blank=True)
    mkt_tsm_adr_gap = models.FloatField(null=True, blank=True)
    mkt_usdtwd_chg_20d = models.FloatField(null=True, blank=True)
    mkt_business_signal = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "feature_daily"
        indexes = [models.Index(fields=["date"], name="feature_daily_date_idx")]
        constraints = [
            models.CheckConstraint(
                condition=finite(*feature_names(batch=1)), name="feature_daily_finite"
            ),
        ]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.date}"
