from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("market", "0003_revenue_statement"),
    ]

    operations = [
        migrations.CreateModel(
            name="DailyValuation",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "stock_id",
                        "date",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("stock_id", models.CharField(max_length=10)),
                ("date", models.DateField()),
                ("pe", models.FloatField(blank=True, null=True)),
                ("pb", models.FloatField(blank=True, null=True)),
                ("dividend_yield", models.FloatField(blank=True, null=True)),
            ],
            options={
                "db_table": "daily_valuation",
                "indexes": [models.Index(fields=["date"], name="daily_valuation_date_idx")],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(
                            ("pb__gt", 0), ("pe__gt", 0), ("dividend_yield__gte", 0)
                        ),
                        name="daily_valuation_ranges",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("pe__gt", float("-inf")),
                            ("pe__lt", float("inf")),
                            ("pb__gt", float("-inf")),
                            ("pb__lt", float("inf")),
                            ("dividend_yield__gt", float("-inf")),
                            ("dividend_yield__lt", float("inf")),
                        ),
                        name="daily_valuation_finite",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="DispositionEvent",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "stock_id",
                        "start_date",
                        "type",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("stock_id", models.CharField(max_length=10)),
                ("start_date", models.DateField()),
                (
                    "type",
                    models.CharField(
                        choices=[
                            ("attention", "注意"),
                            ("disposition", "處置"),
                            ("full_delivery", "全額交割"),
                        ],
                        max_length=16,
                    ),
                ),
                ("end_date", models.DateField(blank=True, null=True)),
            ],
            options={
                "db_table": "disposition_event",
                "indexes": [models.Index(fields=["start_date"], name="disposition_start_idx")],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(
                            ("type__in", ["attention", "disposition", "full_delivery"])
                        ),
                        name="disposition_event_type_valid",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("end_date__isnull", True),
                            ("end_date__gte", models.F("start_date")),
                            _connector="OR",
                        ),
                        name="disposition_event_end_after_start",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="IndexDaily",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "index_id",
                        "date",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "index_id",
                    models.CharField(
                        choices=[
                            ("TAIEX", "加權指數"),
                            ("TPEX", "櫃買指數"),
                            ("SOX", "費城半導體指數"),
                            ("TSM_ADR", "台積電 ADR"),
                            ("USDTWD", "美元兌台幣"),
                        ],
                        max_length=16,
                    ),
                ),
                ("date", models.DateField()),
                ("close", models.DecimalField(decimal_places=4, max_digits=16)),
                ("volume", models.BigIntegerField(blank=True, null=True)),
            ],
            options={
                "db_table": "index_daily",
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(
                            ("index_id__in", ["TAIEX", "TPEX", "SOX", "TSM_ADR", "USDTWD"])
                        ),
                        name="index_daily_index_id_valid",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("close__gt", 0)), name="index_daily_close_positive"
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("volume__gte", 0)), name="index_daily_volume_gte_0"
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="MacroIndicator",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "indicator_id",
                        "period",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("indicator_id", models.CharField(max_length=32)),
                ("period", models.CharField(max_length=7, verbose_name="期間（YYYY-MM）")),
                ("value", models.FloatField()),
                ("announce_date", models.DateField()),
            ],
            options={
                "db_table": "macro_indicator",
                "indexes": [
                    models.Index(fields=["announce_date"], name="macro_indicator_announce_idx")
                ],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("period__regex", "^[0-9]{4}-(0[1-9]|1[0-2])$")),
                        name="macro_indicator_period_format",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("value__gt", float("-inf")), ("value__lt", float("inf"))
                        ),
                        name="macro_indicator_value_finite",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="SharesOutstanding",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "stock_id",
                        "effective_date",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("stock_id", models.CharField(max_length=10)),
                ("effective_date", models.DateField()),
                ("shares", models.BigIntegerField(verbose_name="流通股數（股）")),
                (
                    "reason",
                    models.CharField(
                        choices=[
                            ("unknown", "未知"),
                            ("capital_increase", "增資"),
                            ("capital_reduction", "減資"),
                            ("treasury_cancellation", "庫藏股註銷"),
                        ],
                        default="unknown",
                        max_length=32,
                    ),
                ),
            ],
            options={
                "db_table": "shares_outstanding",
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("shares__gt", 0)),
                        name="shares_outstanding_shares_positive",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            (
                                "reason__in",
                                [
                                    "unknown",
                                    "capital_increase",
                                    "capital_reduction",
                                    "treasury_cancellation",
                                ],
                            )
                        ),
                        name="shares_outstanding_reason_valid",
                    ),
                ],
            },
        ),
    ]
