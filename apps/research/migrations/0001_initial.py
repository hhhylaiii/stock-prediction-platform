import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="HoldoutRun",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                (
                    "scope",
                    models.CharField(
                        choices=[
                            ("rank", "排名"),
                            ("return", "絕對報酬"),
                            ("interval", "區間與機率"),
                        ],
                        max_length=16,
                    ),
                ),
                ("attempt", models.PositiveSmallIntegerField()),
                ("run_key", models.CharField(max_length=64)),
                ("last_date", models.DateField()),
                ("reason", models.TextField(blank=True, default="")),
                ("claimed_at", models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                "db_table": "holdout_run",
                "constraints": [
                    models.UniqueConstraint(
                        fields=("scope", "attempt"), name="holdout_run_scope_attempt"
                    ),
                    models.UniqueConstraint(fields=("run_key",), name="holdout_run_run_key"),
                    models.CheckConstraint(
                        condition=models.Q(("scope__in", ["rank", "return", "interval"])),
                        name="holdout_run_scope_known",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("attempt__gte", 1)), name="holdout_run_attempt_gte_1"
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("attempt", 1), ("reason__regex", "\\S"), _connector="OR"
                        ),
                        name="holdout_run_rerun_has_reason",
                    ),
                ],
            },
        ),
    ]
