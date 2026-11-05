from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0008_featuredaily_fund_rev_yoy_q"),
    ]

    operations = [
        migrations.AddIndex(
            model_name="prediction",
            index=models.Index(
                fields=["run", "horizon", "date"], name="prediction_run_horizon_date"
            ),
        ),
    ]
