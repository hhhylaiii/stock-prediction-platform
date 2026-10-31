from django.apps import AppConfig


class WebConfig(AppConfig):
    """Single-page web views served from Django templates."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.web"
    label = "web"
