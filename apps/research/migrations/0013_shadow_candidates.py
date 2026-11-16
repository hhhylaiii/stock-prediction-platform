import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0012_holding_books"),
    ]

    operations = [
        migrations.CreateModel(
            name="ShadowArtifact",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "candidate",
                        "serving_run_id",
                        "generation",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "candidate",
                    models.CharField(
                        choices=[("C1", "Ridge 與 C0 的排名平均"), ("C2", "5 日相對大盤報酬")],
                        max_length=4,
                    ),
                ),
                ("generation", models.PositiveSmallIntegerField()),
                ("as_of", models.DateField()),
                ("train_start", models.DateField()),
                ("train_end", models.DateField()),
                ("fold", models.JSONField()),
                ("path", models.CharField(max_length=255)),
                ("sha256", models.CharField(max_length=64)),
                ("recipe", models.JSONField()),
                ("first_signal", models.DateField()),
                ("frozen_at", models.DateTimeField(default=django.utils.timezone.now)),
                (
                    "serving_run",
                    models.ForeignKey(
                        db_index=False,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="shadow_artifacts",
                        to="research.modelrun",
                    ),
                ),
            ],
            options={
                "db_table": "shadow_artifact",
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(("generation__gte", 1)),
                        name="shadow_artifact_generation",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("sha256__regex", "^[0-9a-f]{64}$")),
                        name="shadow_artifact_sha256",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("train_start__lt", models.F("train_end"))),
                        name="shadow_artifact_window",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("first_signal__gt", models.F("as_of"))),
                        name="shadow_artifact_scores_after_as_of",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="ShadowScore",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "candidate",
                        "date",
                        "stock_id",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "candidate",
                    models.CharField(
                        choices=[("C1", "Ridge 與 C0 的排名平均"), ("C2", "5 日相對大盤報酬")],
                        max_length=4,
                    ),
                ),
                ("date", models.DateField()),
                ("stock_id", models.CharField(max_length=10)),
                ("generation", models.PositiveSmallIntegerField()),
                ("score", models.FloatField(blank=True, null=True)),
                ("raw", models.FloatField(blank=True, null=True)),
                ("pct", models.FloatField(blank=True, null=True)),
                ("in_universe", models.BooleanField()),
                ("recorded_at", models.DateTimeField(default=django.utils.timezone.now)),
                (
                    "serving_run",
                    models.ForeignKey(
                        db_index=False,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="shadow_scores",
                        to="research.modelrun",
                    ),
                ),
            ],
            options={
                "db_table": "shadow_score",
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(
                            ("score__gt", float("-inf")),
                            ("score__lt", float("inf")),
                            ("raw__gt", float("-inf")),
                            ("raw__lt", float("inf")),
                            ("pct__gt", float("-inf")),
                            ("pct__lt", float("inf")),
                        ),
                        name="shadow_score_finite",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            models.Q(("pct__gte", 0), ("pct__lte", 1)),
                            ("pct__isnull", True),
                            _connector="OR",
                        ),
                        name="shadow_score_pct_unit",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("pct__isnull", True),
                            models.Q(("in_universe", True), ("score__isnull", False)),
                            _connector="OR",
                        ),
                        name="shadow_score_pct_in_universe",
                    ),
                ],
            },
        ),
    ]
