from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("etl", "0005_raw_mops"),
    ]

    operations = [
        migrations.CreateModel(
            name="DividendNotice",
            fields=[
                ("id", models.BigAutoField(primary_key=True, serialize=False)),
                ("source", models.CharField(max_length=32)),
                ("stock_id", models.CharField(max_length=10)),
                ("ex_date", models.DateField()),
                ("first_seen", models.DateField()),
                ("last_seen", models.DateField()),
                ("created_event", models.BooleanField(default=False)),
                ("withdrawn", models.DateField(blank=True, null=True)),
            ],
            options={
                "db_table": "dividend_notice",
                "indexes": [
                    models.Index(fields=["source", "last_seen"], name="dividend_notice_seen_idx")
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("source", "stock_id", "ex_date"), name="dividend_notice_event_uniq"
                    )
                ],
            },
        ),
    ]
