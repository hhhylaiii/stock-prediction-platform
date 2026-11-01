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
                ("scope", models.CharField(max_length=64)),
                ("attempt", models.PositiveSmallIntegerField()),
                ("run_key", models.CharField(max_length=64)),
                ("reason", models.TextField(blank=True, default="")),
                ("claimed_at", models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                "db_table": "holdout_run",
                "constraints": [
                    models.UniqueConstraint(
                        fields=("scope", "attempt"), name="holdout_run_scope_attempt"
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
