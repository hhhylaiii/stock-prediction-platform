"""Source adapter registry used by the ETL management commands."""

from functools import cache

from django.conf import settings

from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.finmind import FinmindAdapter
from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.adapters.twse import TwseAdapter
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.market.calendar import TradingDays

ADAPTERS: dict[str, type[BaseAdapter]] = {
    TwseAdapter.source: TwseAdapter,
    TpexAdapter.source: TpexAdapter,
    FinmindAdapter.source: FinmindAdapter,
}


@cache
def limiter_for(source: str) -> RateLimiter:
    """One limiter per source per process, so back-to-back adapters still keep the interval."""
    return RateLimiter(RateLimit.from_settings(settings.ETL_RATE_LIMITS[source]))


def build_adapter(source: str, calendar: TradingDays | None = None) -> BaseAdapter:
    """Instantiate a registered adapter with its source's shared limiter and the DB calendar."""
    try:
        adapter_class = ADAPTERS[source]
    except KeyError:
        raise ValueError(f"unknown source {source!r}; choose from {sorted(ADAPTERS)}") from None
    return adapter_class(calendar or TradingDays.from_db(), limiter_for(source))
