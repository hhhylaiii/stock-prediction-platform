from django.apps import AppConfig


class EtlConfig(AppConfig):
    """Source adapters, raw layer, ETL runs and data quality."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.etl"
    label = "etl"

    def ready(self) -> None:
        """Hook the task-failure alert; this only registers a signal receiver."""
        from apps.etl import task_alerts

        task_alerts.connect()
