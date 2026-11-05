"""Research-layer tables."""

from django.db import models
from django.utils import timezone

from apps.market.models import finite
from apps.research.features.registry import feature_names

OUTPUT_HORIZONS = [5, 20]
REST_FEATURE = "_rest"


def horizon_known(table: str) -> models.CheckConstraint:
    """The horizon is one the labels carry."""
    return models.CheckConstraint(
        condition=models.Q(horizon__in=OUTPUT_HORIZONS), name=f"{table}_horizon_known"
    )


def unit_interval(*fields: str) -> models.Q:
    """Each column is NULL or within [0, 1]."""
    condition = models.Q()
    for field in fields:
        condition &= models.Q(**{f"{field}__gte": 0, f"{field}__lte": 1}) | models.Q(
            **{f"{field}__isnull": True}
        )
    return condition


def ordered(low: str, high: str) -> models.Q:
    """low <= high whenever both are present."""
    return (
        models.Q(**{f"{low}__lte": models.F(high)})
        | models.Q(**{f"{low}__isnull": True})
        | models.Q(**{f"{high}__isnull": True})
    )


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


class Prediction(models.Model):
    """One run's outputs per stock, day and horizon; columns the run does not produce stay NULL."""

    pk = models.CompositePrimaryKey("run_id", "stock_id", "date", "horizon")
    run = models.ForeignKey(
        ModelRun, on_delete=models.PROTECT, db_index=False, related_name="predictions"
    )
    stock_id = models.CharField(max_length=10)
    date = models.DateField()
    horizon = models.PositiveSmallIntegerField()
    score = models.FloatField(null=True, blank=True)
    rank = models.PositiveIntegerField(null=True, blank=True)
    pct = models.FloatField(null=True, blank=True)
    ret_pred = models.FloatField(null=True, blank=True)
    q10 = models.FloatField(null=True, blank=True)
    q50 = models.FloatField(null=True, blank=True)
    q90 = models.FloatField(null=True, blank=True)
    vol_pred = models.FloatField(null=True, blank=True)
    prob_up_5pct = models.FloatField(null=True, blank=True)
    in_universe = models.BooleanField()

    class Meta:
        db_table = "prediction"
        indexes = [
            models.Index(fields=["date", "horizon", "rank"], name="prediction_date_horizon_rank"),
            models.Index(fields=["stock_id", "date"], name="prediction_stock_date_idx"),
        ]
        constraints = [
            horizon_known("prediction"),
            models.CheckConstraint(
                condition=finite(
                    "score", "pct", "ret_pred", "q10", "q50", "q90", "vol_pred", "prob_up_5pct"
                ),
                name="prediction_finite",
            ),
            models.CheckConstraint(
                condition=models.Q(rank__gte=1) | models.Q(rank__isnull=True),
                name="prediction_rank_gte_1",
            ),
            models.CheckConstraint(
                condition=models.Q(rank__isnull=True, pct__isnull=True)
                | models.Q(rank__isnull=False, pct__isnull=False, score__isnull=False),
                name="prediction_rank_pct_with_score",
            ),
            models.CheckConstraint(
                condition=unit_interval("pct", "prob_up_5pct"), name="prediction_unit_interval"
            ),
            models.CheckConstraint(
                condition=models.Q(vol_pred__gte=0) | models.Q(vol_pred__isnull=True),
                name="prediction_vol_gte_0",
            ),
            models.CheckConstraint(
                condition=ordered("q10", "q50") & ordered("q50", "q90") & ordered("q10", "q90"),
                name="prediction_quantiles_ordered",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.run_id} {self.stock_id} {self.date} {self.horizon}d"


class PredictionContrib(models.Model):
    """Top feature contributions to a rank prediction, then one row summing the rest (K3)."""

    pk = models.CompositePrimaryKey("run_id", "stock_id", "date", "horizon", "feature")
    run = models.ForeignKey(
        ModelRun, on_delete=models.PROTECT, db_index=False, related_name="contributions"
    )
    stock_id = models.CharField(max_length=10)
    date = models.DateField()
    horizon = models.PositiveSmallIntegerField()
    feature = models.CharField(max_length=64)
    position = models.PositiveSmallIntegerField()
    contrib = models.FloatField()
    n_features = models.PositiveSmallIntegerField(default=1)

    class Meta:
        db_table = "prediction_contrib"
        indexes = [models.Index(fields=["stock_id", "date"], name="prediction_contrib_stock_date")]
        constraints = [
            horizon_known("prediction_contrib"),
            models.UniqueConstraint(
                fields=["run_id", "stock_id", "date", "horizon", "position"],
                name="prediction_contrib_position_unique",
            ),
            models.CheckConstraint(condition=finite("contrib"), name="prediction_contrib_finite"),
            models.CheckConstraint(
                condition=models.Q(position__gte=1), name="prediction_contrib_position_gte_1"
            ),
            models.CheckConstraint(
                condition=models.Q(n_features=1)
                | models.Q(feature=REST_FEATURE, n_features__gte=1),
                name="prediction_contrib_rest_sums_features",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.run_id} {self.stock_id} {self.date} {self.horizon}d {self.feature}"


class ModelMonitor(models.Model):
    """Matured-prediction metrics of one run per prediction date and horizon (SPEC §7)."""

    pk = models.CompositePrimaryKey("run_id", "date", "horizon")
    run = models.ForeignKey(
        ModelRun, on_delete=models.PROTECT, db_index=False, related_name="monitors"
    )
    date = models.DateField()
    horizon = models.PositiveSmallIntegerField()
    ic = models.FloatField(null=True, blank=True)
    coverage = models.FloatField(null=True, blank=True)
    brier = models.FloatField(null=True, blank=True)
    direction_acc = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    width = models.FloatField(null=True, blank=True)
    auc = models.FloatField(null=True, blank=True)
    n_stocks = models.PositiveIntegerField()

    class Meta:
        db_table = "model_monitor"
        constraints = [
            horizon_known("model_monitor"),
            models.CheckConstraint(
                condition=finite("ic", "coverage", "brier", "direction_acc", "mae", "width", "auc"),
                name="model_monitor_finite",
            ),
            models.CheckConstraint(
                condition=models.Q(ic__gte=-1, ic__lte=1) | models.Q(ic__isnull=True),
                name="model_monitor_ic_range",
            ),
            models.CheckConstraint(
                condition=unit_interval("coverage", "brier", "direction_acc", "auc"),
                name="model_monitor_unit_interval",
            ),
            models.CheckConstraint(
                condition=(models.Q(mae__gte=0) | models.Q(mae__isnull=True))
                & (models.Q(width__gte=0) | models.Q(width__isnull=True)),
                name="model_monitor_spread_gte_0",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.run_id} {self.date} {self.horizon}d"


class MetricGrain(models.TextChoices):
    TOTAL = "total", "全期"
    FOLD = "fold", "每摺"
    YEAR = "year", "每年"
    MONTH = "month", "每月"


class BacktestMetric(models.Model):
    """One walk-forward metric of a run's model over a period, optionally for a segment."""

    run_key = models.CharField(max_length=64)
    model_type = models.CharField(max_length=32)
    objective = models.CharField(max_length=16, choices=ModelObjective.choices)
    horizon = models.PositiveSmallIntegerField()
    metric = models.CharField(max_length=32)
    grain = models.CharField(max_length=8, choices=MetricGrain.choices)
    period_start = models.DateField()
    period_end = models.DateField()
    segment = models.CharField(max_length=32, blank=True, default="")
    value = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "backtest_metric"
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "run_key",
                    "model_type",
                    "objective",
                    "horizon",
                    "metric",
                    "grain",
                    "period_start",
                    "segment",
                ],
                name="backtest_metric_key",
            ),
            models.CheckConstraint(
                condition=models.Q(objective__in=ModelObjective.values),
                name="backtest_metric_objective_known",
            ),
            models.CheckConstraint(
                condition=models.Q(grain__in=MetricGrain.values),
                name="backtest_metric_grain_known",
            ),
            horizon_known("backtest_metric"),
            models.CheckConstraint(
                condition=models.Q(period_start__lte=models.F("period_end")),
                name="backtest_metric_period_ordered",
            ),
            models.CheckConstraint(condition=finite("value"), name="backtest_metric_finite"),
            models.CheckConstraint(
                condition=models.Q(
                    run_key__regex=r"\S", model_type__regex=r"\S", metric__regex=r"\S"
                ),
                name="backtest_metric_names_present",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.run_key} {self.model_type} {self.metric} {self.grain} {self.period_start}"
