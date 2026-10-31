from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("market", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="DailyChip",
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
                (
                    "fi_net",
                    models.BigIntegerField(blank=True, null=True, verbose_name="外資買賣超（股）"),
                ),
                (
                    "it_net",
                    models.BigIntegerField(blank=True, null=True, verbose_name="投信買賣超（股）"),
                ),
                (
                    "dealer_self_net",
                    models.BigIntegerField(
                        blank=True, null=True, verbose_name="自營商自行買賣超（股）"
                    ),
                ),
                (
                    "dealer_hedge_net",
                    models.BigIntegerField(
                        blank=True, null=True, verbose_name="自營商避險買賣超（股）"
                    ),
                ),
                (
                    "margin_balance",
                    models.BigIntegerField(blank=True, null=True, verbose_name="融資餘額（股）"),
                ),
                (
                    "margin_limit",
                    models.BigIntegerField(blank=True, null=True, verbose_name="融資限額（股）"),
                ),
                (
                    "short_balance",
                    models.BigIntegerField(blank=True, null=True, verbose_name="融券餘額（股）"),
                ),
                (
                    "sbl_short_balance",
                    models.BigIntegerField(
                        blank=True, null=True, verbose_name="借券賣出餘額（股）"
                    ),
                ),
                (
                    "fi_holding_pct",
                    models.FloatField(blank=True, null=True, verbose_name="外資持股比率（%）"),
                ),
                (
                    "day_trade_volume",
                    models.BigIntegerField(
                        blank=True, null=True, verbose_name="當沖成交股數（股）"
                    ),
                ),
            ],
            options={
                "db_table": "daily_chip",
                "indexes": [models.Index(fields=["date"], name="daily_chip_date_idx")],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(
                            ("margin_balance__gte", 0),
                            ("margin_limit__gte", 0),
                            ("short_balance__gte", 0),
                            ("sbl_short_balance__gte", 0),
                            ("day_trade_volume__gte", 0),
                        ),
                        name="daily_chip_balances_gte_0",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("fi_holding_pct__gte", 0), ("fi_holding_pct__lte", 100)
                        ),
                        name="daily_chip_fi_holding_pct_range",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="DividendEvent",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "stock_id",
                        "ex_date",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("stock_id", models.CharField(max_length=10)),
                ("ex_date", models.DateField()),
                (
                    "cash_dividend",
                    models.DecimalField(
                        decimal_places=8,
                        default=0,
                        max_digits=16,
                        verbose_name="現金股利（元／股）",
                    ),
                ),
                (
                    "stock_dividend",
                    models.DecimalField(
                        decimal_places=8,
                        default=0,
                        max_digits=16,
                        verbose_name="股票股利（元／股，面額 10 元）",
                    ),
                ),
                ("adj_factor", models.FloatField(blank=True, null=True)),
            ],
            options={
                "db_table": "dividend_event",
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("cash_dividend__gte", 0), ("stock_dividend__gte", 0)),
                        name="dividend_event_amounts_gte_0",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("adj_factor__isnull", True),
                            models.Q(("adj_factor__gt", 0), ("adj_factor__lt", float("inf"))),
                            _connector="OR",
                        ),
                        name="dividend_event_adj_factor_finite_positive",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="WeeklyHolding",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "stock_id",
                        "data_date",
                        "level",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("stock_id", models.CharField(max_length=10)),
                ("data_date", models.DateField()),
                ("level", models.PositiveSmallIntegerField()),
                ("holders", models.BigIntegerField(verbose_name="人數")),
                ("shares", models.BigIntegerField(verbose_name="股數（股）")),
                ("pct", models.FloatField(verbose_name="占集保庫存比例（%）")),
                ("publish_date", models.DateField()),
            ],
            options={
                "db_table": "weekly_holding",
                "indexes": [
                    models.Index(fields=["publish_date"], name="weekly_holding_publish_idx")
                ],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("level__gte", 1), ("level__lte", 17)),
                        name="weekly_holding_level_range",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("publish_date__gt", models.F("data_date"))),
                        name="weekly_holding_published_after_data_date",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("level", 16),
                            models.Q(("holders__gte", 0), ("shares__gte", 0), ("pct__gte", 0)),
                            _connector="OR",
                        ),
                        name="weekly_holding_counts_gte_0",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("pct__gt", float("-inf")), ("pct__lt", float("inf"))),
                        name="weekly_holding_pct_finite",
                    ),
                ],
            },
        ),
    ]
