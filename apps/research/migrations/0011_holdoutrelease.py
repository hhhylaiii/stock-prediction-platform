import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0010_prediction_run_horizon_date"),
    ]

    operations = [
        migrations.CreateModel(
            name="HoldoutRelease",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("達標", "Passed"),
                            ("未達標", "Failed"),
                            ("明顯退化", "Degraded"),
                        ],
                        max_length=16,
                    ),
                ),
                ("failed", models.JSONField(default=list)),
                ("flagged", models.JSONField(default=list)),
                ("report_sha256", models.CharField(max_length=64)),
                ("reason", models.TextField()),
                ("released_at", models.DateTimeField(default=django.utils.timezone.now)),
                (
                    "holdout",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="release",
                        to="research.holdoutrun",
                    ),
                ),
            ],
            options={
                "db_table": "holdout_release",
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("status__in", ["未達標", "明顯退化"])),
                        name="holdout_release_status_short",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("reason__regex", "\\S")),
                        name="holdout_release_has_reason",
                    ),
                ],
            },
        ),
    ]
