from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("etl", "0003_raw_tdcc_yahoo_datagov"),
    ]

    operations = [
        migrations.CreateModel(
            name="DataQualityLog",
            fields=[
                ("id", models.BigAutoField(primary_key=True, serialize=False)),
                ("date", models.DateField()),
                ("check_name", models.CharField(max_length=64)),
                (
                    "severity",
                    models.CharField(
                        choices=[("info", "Info"), ("warning", "Warning"), ("error", "Error")],
                        max_length=16,
                    ),
                ),
                ("detail", models.JSONField(default=dict)),
            ],
            options={
                "db_table": "data_quality_log",
                "indexes": [
                    models.Index(fields=["date", "severity"], name="dq_log_date_severity_idx")
                ],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("severity__in", ["info", "warning", "error"])),
                        name="data_quality_log_severity_valid",
                    )
                ],
            },
        ),
    ]
