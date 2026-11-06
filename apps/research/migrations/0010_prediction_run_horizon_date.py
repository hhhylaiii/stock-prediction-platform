from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0009_featuredaily_second_batch"),
    ]

    operations = [
        migrations.AddIndex(
            model_name="prediction",
            index=models.Index(
                fields=["run", "horizon", "date"], name="prediction_run_horizon_date"
            ),
        ),
    ]
