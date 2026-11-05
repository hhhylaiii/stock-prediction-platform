import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0006_modelmonitor"),
    ]

    operations = [
        migrations.CreateModel(
            name="BacktestMetric",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                ("run_key", models.CharField(max_length=64)),
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
                ("horizon", models.PositiveSmallIntegerField()),
                ("metric", models.CharField(max_length=32)),
                (
                    "grain",
                    models.CharField(
                        choices=[
                            ("total", "全期"),
                            ("fold", "每摺"),
                            ("year", "每年"),
                            ("month", "每月"),
                        ],
                        max_length=8,
                    ),
                ),
                ("period_start", models.DateField()),
                ("period_end", models.DateField()),
                ("segment", models.CharField(blank=True, default="", max_length=32)),
                ("value", models.FloatField(blank=True, null=True)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                "db_table": "backtest_metric",
                "constraints": [
                    models.UniqueConstraint(
                        fields=(
                            "run_key",
                            "model_type",
                            "objective",
                            "horizon",
                            "metric",
                            "grain",
                            "period_start",
                            "period_end",
                            "segment",
                        ),
                        name="backtest_metric_key",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            (
                                "objective__in",
                                ["rank", "regression", "quantile", "volatility", "classification"],
                            )
                        ),
                        name="backtest_metric_objective_known",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("grain__in", ["total", "fold", "year", "month"])),
                        name="backtest_metric_grain_known",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("horizon__in", [5, 20])),
                        name="backtest_metric_horizon_known",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("period_start__lte", models.F("period_end"))),
                        name="backtest_metric_period_ordered",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("value__gt", float("-inf")), ("value__lt", float("inf"))
                        ),
                        name="backtest_metric_finite",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("metric__regex", "\\S"),
                            ("model_type__regex", "\\S"),
                            ("run_key__regex", "\\S"),
                        ),
                        name="backtest_metric_names_present",
                    ),
                ],
            },
        ),
    ]
