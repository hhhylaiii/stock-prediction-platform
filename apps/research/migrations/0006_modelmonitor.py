import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0005_prediction"),
    ]

    operations = [
        migrations.CreateModel(
            name="ModelMonitor",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "run_id",
                        "date",
                        "horizon",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("date", models.DateField()),
                ("horizon", models.PositiveSmallIntegerField()),
                ("ic", models.FloatField(blank=True, null=True)),
                ("coverage", models.FloatField(blank=True, null=True)),
                ("brier", models.FloatField(blank=True, null=True)),
                ("direction_acc", models.FloatField(blank=True, null=True)),
                ("mae", models.FloatField(blank=True, null=True)),
                ("width", models.FloatField(blank=True, null=True)),
                ("auc", models.FloatField(blank=True, null=True)),
                ("n_stocks", models.PositiveIntegerField()),
                (
                    "run",
                    models.ForeignKey(
                        db_index=False,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="monitors",
                        to="research.modelrun",
                    ),
                ),
            ],
            options={
                "db_table": "model_monitor",
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("horizon__in", [5, 20])),
                        name="model_monitor_horizon_known",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("ic__gt", float("-inf")),
                            ("ic__lt", float("inf")),
                            ("coverage__gt", float("-inf")),
                            ("coverage__lt", float("inf")),
                            ("brier__gt", float("-inf")),
                            ("brier__lt", float("inf")),
                            ("direction_acc__gt", float("-inf")),
                            ("direction_acc__lt", float("inf")),
                            ("mae__gt", float("-inf")),
                            ("mae__lt", float("inf")),
                            ("width__gt", float("-inf")),
                            ("width__lt", float("inf")),
                            ("auc__gt", float("-inf")),
                            ("auc__lt", float("inf")),
                        ),
                        name="model_monitor_finite",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            models.Q(("ic__gte", -1), ("ic__lte", 1)),
                            ("ic__isnull", True),
                            _connector="OR",
                        ),
                        name="model_monitor_ic_range",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            models.Q(
                                models.Q(("coverage__gte", 0), ("coverage__lte", 1)),
                                ("coverage__isnull", True),
                                _connector="OR",
                            ),
                            models.Q(
                                models.Q(("brier__gte", 0), ("brier__lte", 1)),
                                ("brier__isnull", True),
                                _connector="OR",
                            ),
                            models.Q(
                                models.Q(("direction_acc__gte", 0), ("direction_acc__lte", 1)),
                                ("direction_acc__isnull", True),
                                _connector="OR",
                            ),
                            models.Q(
                                models.Q(("auc__gte", 0), ("auc__lte", 1)),
                                ("auc__isnull", True),
                                _connector="OR",
                            ),
                        ),
                        name="model_monitor_unit_interval",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            models.Q(("mae__gte", 0), ("mae__isnull", True), _connector="OR"),
                            models.Q(("width__gte", 0), ("width__isnull", True), _connector="OR"),
                        ),
                        name="model_monitor_spread_gte_0",
                    ),
                ],
            },
        ),
    ]
