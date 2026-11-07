from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("market", "0002_remaining_standard_tables"),
    ]

    operations = [
        migrations.RemoveConstraint(
            model_name="indexdaily",
            name="index_daily_index_id_valid",
        ),
        migrations.AlterField(
            model_name="indexdaily",
            name="index_id",
            field=models.CharField(
                choices=[
                    ("TAIEX", "加權指數"),
                    ("TAIEX_TR", "發行量加權股價報酬指數"),
                    ("TPEX", "櫃買指數"),
                    ("SOX", "費城半導體指數"),
                    ("TSM_ADR", "台積電 ADR"),
                    ("USDTWD", "美元兌台幣"),
                ],
                max_length=16,
            ),
        ),
        migrations.AddConstraint(
            model_name="indexdaily",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    ("index_id__in", ["TAIEX", "TAIEX_TR", "TPEX", "SOX", "TSM_ADR", "USDTWD"])
                ),
                name="index_daily_index_id_valid",
            ),
        ),
    ]
