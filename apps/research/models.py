"""Research-layer tables."""

from django.db import models
from django.utils import timezone


class HoldoutRun(models.Model):
    """One claimed holdout run per scope and attempt; attempts after the first carry a reason."""

    scope = models.CharField(max_length=64)
    attempt = models.PositiveSmallIntegerField()
    run_key = models.CharField(max_length=64)
    reason = models.TextField(blank=True, default="")
    claimed_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "holdout_run"
        constraints = [
            models.UniqueConstraint(fields=["scope", "attempt"], name="holdout_run_scope_attempt"),
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
