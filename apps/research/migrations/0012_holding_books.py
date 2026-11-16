import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0011_holdoutrelease"),
    ]

    operations = [
        migrations.CreateModel(
            name="HoldingBookAction",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "book",
                        "signal_date",
                        "stock_id",
                        "phase",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("book", models.CharField(max_length=32)),
                ("signal_date", models.DateField()),
                ("trade_date", models.DateField()),
                ("stock_id", models.CharField(max_length=10)),
                (
                    "phase",
                    models.CharField(choices=[("list", "名單"), ("fill", "成交")], max_length=4),
                ),
                (
                    "action",
                    models.CharField(
                        choices=[
                            ("sell", "賣出"),
                            ("keep", "續抱"),
                            ("buy", "買進"),
                            ("backup", "備位"),
                            ("settle", "結算"),
                            ("stuck", "賣不掉"),
                        ],
                        max_length=8,
                    ),
                ),
                ("priority", models.PositiveIntegerField(default=0)),
                ("score", models.FloatField(blank=True, null=True)),
                ("pct", models.FloatField(blank=True, null=True)),
                ("run_id", models.CharField(blank=True, default="", max_length=128)),
                ("amount", models.FloatField(blank=True, null=True)),
                ("value", models.FloatField(blank=True, null=True)),
                ("cost", models.FloatField(blank=True, null=True)),
                ("limit_down_open", models.BooleanField(blank=True, null=True)),
                ("recorded_at", models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                "db_table": "holding_book_action",
                "indexes": [
                    models.Index(
                        fields=["book", "trade_date", "phase"], name="holding_book_action_trade"
                    )
                ],
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(
                            ("score__gt", float("-inf")),
                            ("score__lt", float("inf")),
                            ("pct__gt", float("-inf")),
                            ("pct__lt", float("inf")),
                            ("amount__gt", float("-inf")),
                            ("amount__lt", float("inf")),
                            ("value__gt", float("-inf")),
                            ("value__lt", float("inf")),
                            ("cost__gt", float("-inf")),
                            ("cost__lt", float("inf")),
                        ),
                        name="holding_book_action_finite",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            models.Q(("pct__gte", 0), ("pct__lte", 1)),
                            ("pct__isnull", True),
                            _connector="OR",
                        ),
                        name="holding_book_action_pct_unit",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="HoldingBookDay",
            fields=[
                (
                    "pk",
                    models.CompositePrimaryKey(
                        "book",
                        "signal_date",
                        "phase",
                        blank=True,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("book", models.CharField(max_length=32)),
                ("signal_date", models.DateField()),
                (
                    "phase",
                    models.CharField(choices=[("list", "名單"), ("fill", "成交")], max_length=4),
                ),
                ("trade_date", models.DateField()),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("listed", "有名單"),
                            ("no_prediction", "當天無預測"),
                            ("traded", "照名單結算"),
                            ("idle", "只估值"),
                        ],
                        max_length=16,
                    ),
                ),
                ("run_id", models.CharField(blank=True, default="", max_length=128)),
                ("universe", models.JSONField(default=list)),
                ("held_before", models.PositiveIntegerField(blank=True, null=True)),
                ("sells", models.PositiveIntegerField(blank=True, null=True)),
                ("buys", models.PositiveIntegerField(blank=True, null=True)),
                ("backups", models.PositiveIntegerField(blank=True, null=True)),
                ("value", models.FloatField(blank=True, null=True)),
                ("cash", models.FloatField(blank=True, null=True)),
                ("paid", models.FloatField(blank=True, null=True)),
                ("traded", models.FloatField(blank=True, null=True)),
                ("held", models.PositiveIntegerField(blank=True, null=True)),
                ("universe_return", models.FloatField(blank=True, null=True)),
                ("index_return", models.FloatField(blank=True, null=True)),
                ("backfilled", models.BooleanField(default=False)),
                ("recorded_at", models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                "db_table": "holding_book_day",
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(
                            models.Q(
                                ("phase", "list"), ("status__in", ["listed", "no_prediction"])
                            ),
                            models.Q(("phase", "fill"), ("status__in", ["traded", "idle"])),
                            _connector="OR",
                        ),
                        name="holding_book_day_status_of_phase",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(("trade_date__gt", models.F("signal_date"))),
                        name="holding_book_day_trades_after_signal",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            models.Q(("held_before__isnull", False), ("phase", "list")),
                            models.Q(
                                ("cash__isnull", False),
                                ("held__isnull", False),
                                ("paid__isnull", False),
                                ("phase", "fill"),
                                ("traded__isnull", False),
                                ("value__isnull", False),
                            ),
                            _connector="OR",
                        ),
                        name="holding_book_day_phase_fields",
                    ),
                    models.CheckConstraint(
                        condition=models.Q(
                            ("value__gt", float("-inf")),
                            ("value__lt", float("inf")),
                            ("cash__gt", float("-inf")),
                            ("cash__lt", float("inf")),
                            ("paid__gt", float("-inf")),
                            ("paid__lt", float("inf")),
                            ("traded__gt", float("-inf")),
                            ("traded__lt", float("inf")),
                            ("universe_return__gt", float("-inf")),
                            ("universe_return__lt", float("inf")),
                            ("index_return__gt", float("-inf")),
                            ("index_return__lt", float("inf")),
                        ),
                        name="holding_book_day_finite",
                    ),
                ],
            },
        ),
    ]
