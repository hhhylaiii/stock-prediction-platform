from django.apps import AppConfig


class ResearchConfig(AppConfig):
    """Features, labels, models and walk-forward evaluation."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.research"
    label = "research"
