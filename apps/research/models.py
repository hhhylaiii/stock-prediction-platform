"""Research-layer tables."""

from django.db import models
from django.utils import timezone

from apps.market.models import finite


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
