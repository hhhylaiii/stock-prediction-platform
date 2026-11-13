from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("market", "0004_market_history"),
    ]

    operations = [
        migrations.AddField(
            model_name="tradingcalendar",
            name="is_provisional",
            field=models.BooleanField(db_default=False, default=False),
        ),
    ]
