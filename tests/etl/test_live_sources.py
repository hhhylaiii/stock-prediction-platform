import logging
import os
from datetime import date, timedelta

import pytest
from django.utils import timezone

from apps.etl.adapters import build_adapter
from apps.market.calendar import TradingDays
from apps.market.models import IndexDaily, MacroIndicator, WeeklyHolding

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.live,
    pytest.mark.django_db,
    pytest.mark.skipif(
        os.environ.get("ETL_LIVE_TESTS") != "1", reason="set ETL_LIVE_TESTS=1 to call real sources"
    ),
]

HISTORY_START = date(2010, 1, 4)


def _weekdays(start: date, end: date) -> TradingDays:
    """A weekday stand-in calendar; these checks only need sources to answer on sessions."""
    days = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return TradingDays((day for day in days if day.weekday() < 5), start, end)


def test_tdcc_serves_the_latest_week() -> None:
    today = timezone.localdate()
    adapter = build_adapter("tdcc", _weekdays(today - timedelta(days=14), today))
    run = adapter.run("weekly_holding", today - timedelta(days=7), today)
    codes = WeeklyHolding.objects.values("stock_id").distinct().count()
    logger.info("tdcc weekly_holding: %d rows, %d codes", run.rows, codes)
    assert codes > 4000


@pytest.mark.parametrize("dataset", ["sox", "tsm_adr", "usdtwd"])
def test_yahoo_serves_history_since_2010(dataset: str) -> None:
    today = timezone.localdate()
    adapter = build_adapter("yahoo", _weekdays(HISTORY_START, today))
    run = adapter.run(dataset, HISTORY_START, today - timedelta(days=1))
    first = IndexDaily.objects.order_by("date").values_list("date", flat=True).first()
    logger.info("yahoo %s: %d rows from %s", dataset, run.rows, first)
    assert run.rows > 4000


def test_datagov_serves_the_business_signal_history() -> None:
    today = timezone.localdate()
    run = build_adapter("datagov", _weekdays(today, today)).run("business_signal", today, today)
    latest = MacroIndicator.objects.order_by("-period").values_list("period", flat=True).first()
    logger.info("datagov business_signal: %d rows up to %s", run.rows, latest)
    assert run.rows > 500
