"""Research-layer tables."""

from django.db import models
from django.utils import timezone


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
