import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="EtlRun",
            fields=[
                ("run_id", models.BigAutoField(primary_key=True, serialize=False)),
                ("source", models.CharField(max_length=32)),
                ("dataset", models.CharField(max_length=64)),
                ("start_date", models.DateField()),
                ("end_date", models.DateField()),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("running", "Running"),
                            ("success", "Success"),
                            ("failed", "Failed"),
                        ],
                        default="running",
                        max_length=16,
                    ),
                ),
                ("rows", models.IntegerField(default=0)),
                ("error", models.TextField(blank=True, default="")),
                ("checkpoint", models.JSONField(default=dict)),
                ("started_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
            ],
            options={
                "db_table": "etl_run",
                "indexes": [
                    models.Index(
                        fields=["source", "dataset", "start_date"],
                        name="etl_run_source_dataset_idx",
                    )
                ],
            },
        ),
        migrations.CreateModel(
            name="RawTwse",
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
                "db_table": "raw_twse",
                "indexes": [
                    models.Index(
                        fields=["dataset", "fetched_at"], name="raw_twse_dataset_fetched_idx"
                    )
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("dataset", "source_hash"), name="raw_twse_dataset_hash_uniq"
                    )
                ],
            },
        ),
    ]
