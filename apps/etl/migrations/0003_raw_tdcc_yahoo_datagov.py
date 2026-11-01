import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("etl", "0002_raw_finmind_tpex"),
    ]

    operations = [
        migrations.CreateModel(
            name="RawDatagov",
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
                "db_table": "raw_datagov",
                "indexes": [
                    models.Index(
                        fields=["dataset", "fetched_at"], name="raw_datagov_ds_fetched_idx"
                    )
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("dataset", "source_hash"), name="raw_datagov_dataset_hash_uniq"
                    )
                ],
            },
        ),
        migrations.CreateModel(
            name="RawTdcc",
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
                "db_table": "raw_tdcc",
                "indexes": [
                    models.Index(
                        fields=["dataset", "fetched_at"], name="raw_tdcc_dataset_fetched_idx"
                    )
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("dataset", "source_hash"), name="raw_tdcc_dataset_hash_uniq"
                    )
                ],
            },
        ),
        migrations.CreateModel(
            name="RawYahoo",
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
                "db_table": "raw_yahoo",
                "indexes": [
                    models.Index(
                        fields=["dataset", "fetched_at"], name="raw_yahoo_dataset_fetched_idx"
                    )
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("dataset", "source_hash"), name="raw_yahoo_dataset_hash_uniq"
                    )
                ],
            },
        ),
    ]
