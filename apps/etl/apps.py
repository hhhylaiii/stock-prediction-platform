from django.apps import AppConfig


class EtlConfig(AppConfig):
    """Source adapters, raw layer, ETL runs and data quality."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.etl"
    label = "etl"
