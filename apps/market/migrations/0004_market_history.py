from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("market", "0003_taiex_total_return"),
    ]

    operations = [
        migrations.CreateModel(
            name="MarketHistory",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                ("stock_id", models.CharField(max_length=10)),
                ("start_date", models.DateField()),
                ("end_date", models.DateField(blank=True, null=True)),
                (
                    "market",
                    models.CharField(choices=[("TWSE", "上市"), ("TPEx", "上櫃")], max_length=4),
                ),
            ],
            options={
                "db_table": "market_history",
                "constraints": [
                    models.UniqueConstraint(
                        fields=("stock_id", "start_date"), name="market_history_stock_start_uniq"
                    ),
                    models.UniqueConstraint(
                        condition=models.Q(("end_date__isnull", True)),
                        fields=("stock_id",),
                        name="market_history_one_open_span",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("end_date__isnull", True),
                            ("end_date__gte", models.F("start_date")),
                            _connector="OR",
                        ),
                        name="market_history_end_after_start",
                    ),
                ],
            },
        ),
    ]
