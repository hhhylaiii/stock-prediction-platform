import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0003_featuredaily"),
    ]

    operations = [
        migrations.CreateModel(
            name="ModelRun",
            fields=[
                ("run_id", models.CharField(max_length=64, primary_key=True, serialize=False)),
                ("run_key", models.CharField(db_index=True, max_length=64)),
                ("model_type", models.CharField(max_length=32)),
                (
                    "objective",
                    models.CharField(
                        choices=[
                            ("rank", "排名"),
                            ("regression", "絕對報酬"),
                            ("quantile", "分位數"),
                            ("volatility", "波動率"),
                            ("classification", "上漲機率"),
                        ],
                        max_length=16,
                    ),
                ),
                ("version", models.CharField(max_length=32)),
                ("train_start", models.DateField()),
                ("train_end", models.DateField()),
                ("params", models.JSONField(blank=True, default=dict)),
                ("feature_list", models.JSONField(blank=True, default=list)),
                ("seed", models.IntegerField(blank=True, null=True)),
                ("lib_versions", models.JSONField(default=dict)),
                ("feature_hash", models.CharField(max_length=64)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("candidate", "候選"),
                            ("active", "上線"),
                            ("degraded", "降級"),
                            ("retired", "退役"),
                        ],
                        default="candidate",
                        max_length=16,
                    ),
                ),
                ("ic_mean", models.FloatField(blank=True, null=True)),
                ("icir", models.FloatField(blank=True, null=True)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                "db_table": "model_run",
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(
                            (
                                "objective__in",
                                ["rank", "regression", "quantile", "volatility", "classification"],
                            )
                        ),
                        name="model_run_objective_known",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("status__in", ["candidate", "active", "degraded", "retired"])
                        ),
                        name="model_run_status_known",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("train_start__lte", models.F("train_end"))),
                        name="model_run_window_ordered",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("ic_mean__gt", float("-inf")),
                            ("ic_mean__lt", float("inf")),
                            ("icir__gt", float("-inf")),
                            ("icir__lt", float("inf")),
                        ),
                        name="model_run_finite",
                    ),
                ],
            },
        ),
    ]
