from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("market", "0004_index_macro_shares_valuation_disposition"),
    ]

    operations = [
        migrations.RemoveConstraint(
            model_name="dailyvaluation",
            name="daily_valuation_ranges",
        ),
        migrations.AddField(
            model_name="dividendevent",
            name="announce_date",
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="indexdaily",
            name="volume",
            field=models.BigIntegerField(
                blank=True,
                null=True,
                verbose_name="成交量（台股指數：股；TSM_ADR：ADR 數；其餘留空）",
            ),
        ),
        migrations.AlterField(
            model_name="weeklyholding",
            name="holders",
            field=models.IntegerField(verbose_name="人數"),
        ),
        migrations.AddConstraint(
            model_name="dailyvaluation",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    ("dividend_yield__lt", 1),
                    ("pb__gt", 0),
                    ("pe__gt", 0),
                    ("dividend_yield__gte", 0),
                ),
                name="daily_valuation_ranges",
            ),
        ),
        migrations.AddConstraint(
            model_name="dividendevent",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    ("announce_date__isnull", True),
                    ("announce_date__lte", models.F("ex_date")),
                    _connector="OR",
                ),
                name="dividend_event_announced_by_ex_date",
            ),
        ),
    ]
