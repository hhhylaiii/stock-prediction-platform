from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="LabelDaily",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
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
                ("fwd_ret", models.FloatField(blank=True, null=True)),
                ("fwd_ret_open", models.FloatField(blank=True, null=True)),
                ("fwd_rank_pct", models.FloatField(blank=True, null=True)),
                ("fwd_vol_20d", models.FloatField(blank=True, null=True)),
                ("fwd_up_5pct", models.BooleanField(blank=True, null=True)),
                ("untradable_t1", models.BooleanField(blank=True, null=True)),
            ],
            options={
                "db_table": "label_daily",
                "indexes": [
                    models.Index(fields=["date", "horizon"], name="label_daily_date_horizon_idx")
                ],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("horizon__in", [5, 20])),
                        name="label_daily_horizon_known",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("fwd_ret__gt", float("-inf")),
                            ("fwd_ret__lt", float("inf")),
                            ("fwd_ret_open__gt", float("-inf")),
                            ("fwd_ret_open__lt", float("inf")),
                            ("fwd_vol_20d__gt", float("-inf")),
                            ("fwd_vol_20d__lt", float("inf")),
                        ),
                        name="label_daily_finite",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            models.Q(("fwd_rank_pct__gt", 0), ("fwd_rank_pct__lte", 1)),
                            ("fwd_rank_pct__isnull", True),
                            _connector="OR",
                        ),
                        name="label_daily_rank_pct_range",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("fwd_vol_20d__gte", 0), ("fwd_vol_20d__isnull", True), _connector="OR"
                        ),
                        name="label_daily_vol_gte_0",
                    ),
                ],
            },
        ),
    ]
