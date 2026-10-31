import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("etl", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="RawFinmind",
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
                "db_table": "raw_finmind",
                "indexes": [
                    models.Index(
                        fields=["dataset", "fetched_at"], name="raw_finmind_ds_fetched_idx"
                    )
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("dataset", "source_hash"), name="raw_finmind_dataset_hash_uniq"
                    )
                ],
            },
        ),
        migrations.CreateModel(
            name="RawTpex",
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
                "db_table": "raw_tpex",
                "indexes": [
                    models.Index(
                        fields=["dataset", "fetched_at"], name="raw_tpex_dataset_fetched_idx"
                    )
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("dataset", "source_hash"), name="raw_tpex_dataset_hash_uniq"
                    )
                ],
            },
        ),
    ]
