from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="StockMaster",
            fields=[
                ("stock_id", models.CharField(max_length=10, primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=64)),
                (
                    "market",
                    models.CharField(choices=[("TWSE", "上市"), ("TPEx", "上櫃")], max_length=4),
                ),
                ("industry", models.CharField(blank=True, default="", max_length=64)),
                ("list_date", models.DateField(blank=True, null=True)),
                ("delist_date", models.DateField(blank=True, null=True)),
            ],
            options={
                "db_table": "stock_master",
            },
        ),
        migrations.CreateModel(
            name="TradingCalendar",
            fields=[
                ("date", models.DateField(primary_key=True, serialize=False)),
                ("is_trading_day", models.BooleanField()),
                ("note", models.CharField(blank=True, default="", max_length=64)),
            ],
            options={
                "db_table": "trading_calendar",
            },
        ),
        migrations.CreateModel(
            name="DailyPrice",
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
                    "open",
                    models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True),
                ),
                (
                    "high",
                    models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True),
                ),
                (
                    "low",
                    models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True),
                ),
                (
                    "close",
                    models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True),
                ),
                ("volume", models.BigIntegerField(verbose_name="成交量（股）")),
                ("turnover", models.BigIntegerField(verbose_name="成交金額（元）")),
                ("adj_factor", models.FloatField(default=1.0)),
            ],
            options={
                "db_table": "daily_price",
                "indexes": [models.Index(fields=["date"], name="daily_price_date_idx")],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("volume__gte", 0)), name="daily_price_volume_gte_0"
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("turnover__gte", 0)), name="daily_price_turnover_gte_0"
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("adj_factor__gt", 0), ("adj_factor__lt", float("inf"))),
                        name="daily_price_adj_factor_finite_positive",
                    ),
                ],
            },
        ),
    ]
