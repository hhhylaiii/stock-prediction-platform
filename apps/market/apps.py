from django.apps import AppConfig


class MarketConfig(AppConfig):
    """Standard-layer market data and trading calendar."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.market"
    label = "market"
