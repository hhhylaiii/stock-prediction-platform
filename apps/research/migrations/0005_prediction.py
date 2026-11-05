import django.db.models.constraints
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0004_modelrun"),
    ]

    operations = [
        migrations.CreateModel(
            name="Prediction",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "run_id",
                        "stock_id",
                        "date",
                        "horizon",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("stock_id", models.CharField(max_length=10)),
                ("date", models.DateField()),
                ("horizon", models.PositiveSmallIntegerField()),
                ("score", models.FloatField(blank=True, null=True)),
                ("rank", models.PositiveIntegerField(blank=True, null=True)),
                ("pct", models.FloatField(blank=True, null=True)),
                ("ret_pred", models.FloatField(blank=True, null=True)),
                ("q10", models.FloatField(blank=True, null=True)),
                ("q50", models.FloatField(blank=True, null=True)),
                ("q90", models.FloatField(blank=True, null=True)),
                ("vol_pred", models.FloatField(blank=True, null=True)),
                ("prob_up_5pct", models.FloatField(blank=True, null=True)),
                ("in_universe", models.BooleanField()),
                (
                    "run",
                    models.ForeignKey(
                        db_index=False,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="predictions",
                        to="research.modelrun",
                    ),
                ),
            ],
            options={
                "db_table": "prediction",
                "indexes": [
                    models.Index(
                        fields=["date", "horizon", "rank"], name="prediction_date_horizon_rank"
                    ),
                    models.Index(fields=["stock_id", "date"], name="prediction_stock_date_idx"),
                ],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("horizon__in", [5, 20])),
                        name="prediction_horizon_known",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("score__gt", float("-inf")),
                            ("score__lt", float("inf")),
                            ("pct__gt", float("-inf")),
                            ("pct__lt", float("inf")),
                            ("ret_pred__gt", float("-inf")),
                            ("ret_pred__lt", float("inf")),
                            ("q10__gt", float("-inf")),
                            ("q10__lt", float("inf")),
                            ("q50__gt", float("-inf")),
                            ("q50__lt", float("inf")),
                            ("q90__gt", float("-inf")),
                            ("q90__lt", float("inf")),
                            ("vol_pred__gt", float("-inf")),
                            ("vol_pred__lt", float("inf")),
                            ("prob_up_5pct__gt", float("-inf")),
                            ("prob_up_5pct__lt", float("inf")),
                        ),
                        name="prediction_finite",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("rank__gte", 1), ("rank__isnull", True), _connector="OR"
                        ),
                        name="prediction_rank_gte_1",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            models.Q(("pct__isnull", True), ("rank__isnull", True)),
                            models.Q(
                                ("pct__isnull", False),
                                ("rank__isnull", False),
                                ("score__isnull", False),
                            ),
                            _connector="OR",
                        ),
                        name="prediction_rank_pct_with_score",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            models.Q(
                                models.Q(("pct__gte", 0), ("pct__lte", 1)),
                                ("pct__isnull", True),
                                _connector="OR",
                            ),
                            models.Q(
                                models.Q(("prob_up_5pct__gte", 0), ("prob_up_5pct__lte", 1)),
                                ("prob_up_5pct__isnull", True),
                                _connector="OR",
                            ),
                        ),
                        name="prediction_unit_interval",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("vol_pred__gte", 0), ("vol_pred__isnull", True), _connector="OR"
                        ),
                        name="prediction_vol_gte_0",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            models.Q(
                                ("q10__lte", models.F("q50")),
                                ("q10__isnull", True),
                                ("q50__isnull", True),
                                _connector="OR",
                            ),
                            models.Q(
                                ("q50__lte", models.F("q90")),
                                ("q50__isnull", True),
                                ("q90__isnull", True),
                                _connector="OR",
                            ),
                            models.Q(
                                ("q10__lte", models.F("q90")),
                                ("q10__isnull", True),
                                ("q90__isnull", True),
                                _connector="OR",
                            ),
                        ),
                        name="prediction_quantiles_ordered",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("ret_pred", models.F("q50")),
                            ("ret_pred__isnull", True),
                            ("q50__isnull", True),
                            _connector="OR",
                        ),
                        name="prediction_ret_pred_is_q50",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="PredictionContrib",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "run_id",
                        "stock_id",
                        "date",
                        "horizon",
                        "position",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("stock_id", models.CharField(max_length=10)),
                ("date", models.DateField()),
                ("horizon", models.PositiveSmallIntegerField()),
                ("feature", models.CharField(max_length=64)),
                ("position", models.PositiveSmallIntegerField()),
                ("contrib", models.FloatField()),
                ("n_features", models.PositiveSmallIntegerField(default=1)),
                (
                    "run",
                    models.ForeignKey(
                        db_index=False,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="contributions",
                        to="research.modelrun",
                    ),
                ),
            ],
            options={
                "db_table": "prediction_contrib",
                "indexes": [
                    models.Index(fields=["stock_id", "date"], name="prediction_contrib_stock_date")
                ],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("horizon__in", [5, 20])),
                        name="prediction_contrib_horizon_known",
                    ),
                    models.UniqueConstraint(
                        deferrable=django.db.models.constraints.Deferrable["DEFERRED"],
                        fields=("run_id", "stock_id", "date", "horizon", "feature"),
                        name="prediction_contrib_feature_unique",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("contrib__gt", float("-inf")), ("contrib__lt", float("inf"))
                        ),
                        name="prediction_contrib_finite",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("position__gte", 1)),
                        name="prediction_contrib_position_gte_1",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("n_features", 1),
                            models.Q(("feature", "_rest"), ("n_features__gte", 1)),
                            _connector="OR",
                        ),
                        name="prediction_contrib_rest_sums_features",
                    ),
                ],
            },
        ),
    ]
