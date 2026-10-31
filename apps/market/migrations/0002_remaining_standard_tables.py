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
                            ("dividend_yield__lt", 1),
                            ("pb__gt", 0),
                            ("pe__gt", 0),
                            ("dividend_yield__gte", 0),
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
                (
                    "announce_date",
                    models.DateField(
                        blank=True,
                        null=True,
                        verbose_name="除權息日公告日（ex_date 首次公開的日期）",
                    ),
                ),
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
                    models.CheckConstraint(
                        condition=models.Q(
                            ("announce_date__isnull", True),
                            ("announce_date__lte", models.F("ex_date")),
                            _connector="OR",
                        ),
                        name="dividend_event_announced_by_ex_date",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="FinancialStatement",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "stock_id",
                        "period",
                        "fin_type",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("stock_id", models.CharField(max_length=10)),
                ("period", models.CharField(max_length=7, verbose_name="期間（YYYY-Qn）")),
                (
                    "fin_type",
                    models.CharField(
                        choices=[("consolidated", "合併"), ("individual", "個體")],
                        default="consolidated",
                        max_length=16,
                    ),
                ),
                (
                    "eps",
                    models.DecimalField(
                        blank=True,
                        decimal_places=4,
                        max_digits=12,
                        null=True,
                        verbose_name="EPS（元）",
                    ),
                ),
                (
                    "revenue",
                    models.BigIntegerField(blank=True, null=True, verbose_name="營收（元）"),
                ),
                ("gross_margin", models.FloatField(blank=True, null=True)),
                ("op_margin", models.FloatField(blank=True, null=True)),
                (
                    "net_income",
                    models.BigIntegerField(blank=True, null=True, verbose_name="淨利（元）"),
                ),
                ("roe", models.FloatField(blank=True, null=True)),
                ("debt_ratio", models.FloatField(blank=True, null=True)),
                (
                    "operating_cf",
                    models.BigIntegerField(blank=True, null=True, verbose_name="營業現金流（元）"),
                ),
                ("announce_date", models.DateField()),
            ],
            options={
                "db_table": "financial_statement",
                "indexes": [
                    models.Index(fields=["announce_date"], name="financial_stmt_announce_idx")
                ],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("period__regex", "^[0-9]{4}-Q[1-4]$")),
                        name="financial_statement_period_format",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("fin_type__in", ["consolidated", "individual"])),
                        name="financial_statement_fin_type_valid",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("gross_margin__gt", float("-inf")),
                            ("gross_margin__lt", float("inf")),
                            ("op_margin__gt", float("-inf")),
                            ("op_margin__lt", float("inf")),
                            ("roe__gt", float("-inf")),
                            ("roe__lt", float("inf")),
                            ("debt_ratio__gt", float("-inf")),
                            ("debt_ratio__lt", float("inf")),
                        ),
                        name="financial_statement_finite",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("debt_ratio__gte", 0)),
                        name="financial_statement_debt_ratio_gte_0",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            (
                                "period__lt",
                                models.Func(
                                    models.F("announce_date"),
                                    models.Value('YYYY-"Q"Q'),
                                    function="to_char",
                                    output_field=models.CharField(),
                                ),
                            )
                        ),
                        name="financial_statement_announced_after_period",
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
                (
                    "volume",
                    models.BigIntegerField(
                        blank=True,
                        null=True,
                        verbose_name="成交量（台股指數：股；TSM_ADR：ADR 數；其餘留空）",
                    ),
                ),
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
                    models.CheckConstraint(
                        condition=models.Q(
                            (
                                "period__lt",
                                models.Func(
                                    models.F("announce_date"),
                                    models.Value("YYYY-MM"),
                                    function="to_char",
                                    output_field=models.CharField(),
                                ),
                            )
                        ),
                        name="macro_indicator_announced_after_period",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="MonthlyRevenue",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "stock_id",
                        "period",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("stock_id", models.CharField(max_length=10)),
                ("period", models.CharField(max_length=7, verbose_name="期間（YYYY-MM）")),
                ("revenue", models.BigIntegerField(verbose_name="營收（元）")),
                ("yoy", models.FloatField(blank=True, null=True)),
                ("mom", models.FloatField(blank=True, null=True)),
                ("announce_date", models.DateField()),
            ],
            options={
                "db_table": "monthly_revenue",
                "indexes": [
                    models.Index(fields=["announce_date"], name="monthly_revenue_announce_idx")
                ],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("period__regex", "^[0-9]{4}-(0[1-9]|1[0-2])$")),
                        name="monthly_revenue_period_format",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("yoy__gt", float("-inf")),
                            ("yoy__lt", float("inf")),
                            ("mom__gt", float("-inf")),
                            ("mom__lt", float("inf")),
                        ),
                        name="monthly_revenue_finite",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            (
                                "period__lt",
                                models.Func(
                                    models.F("announce_date"),
                                    models.Value("YYYY-MM"),
                                    function="to_char",
                                    output_field=models.CharField(),
                                ),
                            )
                        ),
                        name="monthly_revenue_announced_after_period",
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
                ("holders", models.IntegerField(verbose_name="人數")),
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
                            models.Q(
                                ("holders__gte", 0),
                                ("shares__gte", 0),
                                ("pct__gte", 0),
                                ("pct__lte", 100),
                            ),
                            _connector="OR",
                        ),
                        name="weekly_holding_counts_in_range",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("pct__gt", float("-inf")), ("pct__lt", float("inf"))),
                        name="weekly_holding_pct_finite",
                    ),
                ],
            },
        ),
    ]
