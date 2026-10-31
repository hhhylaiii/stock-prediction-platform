from django.apps import AppConfig


class ApiConfig(AppConfig):
    """Read-only REST API over predictions and market data."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.api"
    label = "api"
