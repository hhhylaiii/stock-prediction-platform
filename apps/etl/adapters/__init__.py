"""Source adapter registry used by the ETL management commands."""

from django.conf import settings

from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.twse import TwseAdapter
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.market.calendar import TradingDays

ADAPTERS: dict[str, type[BaseAdapter]] = {TwseAdapter.source: TwseAdapter}


def build_adapter(source: str, calendar: TradingDays | None = None) -> BaseAdapter:
    """Instantiate a registered adapter with its configured rate limit and the DB calendar."""
    try:
        adapter_class = ADAPTERS[source]
    except KeyError:
        raise ValueError(f"unknown source {source!r}; choose from {sorted(ADAPTERS)}") from None
    limiter = RateLimiter(RateLimit.from_settings(settings.ETL_RATE_LIMITS[source]))
    return adapter_class(calendar or TradingDays.from_db(), limiter)
