from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("market", "0002_dividend_chip_holding"),
    ]

    operations = [
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
                ],
            },
        ),
    ]
