import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("etl", "0004_data_quality_log"),
    ]

    operations = [
        migrations.CreateModel(
            name="RawMops",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                ("dataset", models.CharField(max_length=64)),
                ("request_params", models.JSONField(default=dict)),
                ("payload", models.JSONField()),
                ("fetched_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("source_hash", models.CharField(max_length=64)),
            ],
            options={
                "db_table": "raw_mops",
                "indexes": [
                    models.Index(fields=["dataset", "fetched_at"], name="raw_mops_ds_fetched_idx")
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("dataset", "source_hash"), name="raw_mops_dataset_hash_uniq"
                    )
                ],
            },
        ),
    ]
