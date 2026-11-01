import logging
import os
from datetime import date, timedelta

import pytest
from django.utils import timezone

from apps.etl.adapters import build_adapter
from apps.market.calendar import TradingDays
from apps.market.models import IndexDaily, MacroIndicator, Market, StockMaster, WeeklyHolding

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.live,
    pytest.mark.django_db,
    pytest.mark.skipif(
        os.environ.get("ETL_LIVE_TESTS") != "1", reason="set ETL_LIVE_TESTS=1 to call real sources"
    ),
]

HISTORY_START = date(2010, 1, 4)
SIGNAL_START = "1984-01"
SATURDAY = 5
LISTED = {"2330": Market.TWSE, "2317": Market.TWSE, "6488": Market.TPEX}
SERIES = {"sox": "SOX", "tsm_adr": "TSM_ADR", "usdtwd": "USDTWD"}


def _weekdays(start: date, end: date) -> TradingDays:
    """A weekday stand-in calendar; these checks only need sources to answer on sessions."""
    days = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return TradingDays((day for day in days if day.weekday() < SATURDAY), start, end)


def test_tdcc_serves_the_latest_week_of_listed_codes() -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code, name=code, market=market) for code, market in LISTED.items()
    )
    today = timezone.localdate()
    weekend = today.weekday() >= SATURDAY
    monday = today - timedelta(days=today.weekday() + (0 if weekend else 7))
    adapter = build_adapter("tdcc", _weekdays(monday, today))
    run = adapter.run("weekly_holding", monday, min(today, monday + timedelta(days=6)))
    week = WeeklyHolding.objects.values_list("data_date", flat=True).first()
    logger.info("tdcc weekly_holding: %d rows for the week of %s", run.rows, week)
    assert run.rows == len(LISTED) * 17


@pytest.mark.parametrize("dataset", sorted(SERIES))
def test_yahoo_serves_history_since_2010(dataset: str) -> None:
    today = timezone.localdate()
    adapter = build_adapter("yahoo", _weekdays(HISTORY_START, today))
    run = adapter.run(dataset, HISTORY_START, today - timedelta(days=1))
    rows = IndexDaily.objects.filter(index_id=SERIES[dataset]).order_by("date")
    first = rows.values_list("date", flat=True).first()
    logger.info("yahoo %s: %d rows from %s", dataset, run.rows, first)
    assert first == HISTORY_START
    assert run.rows > 4000


def test_datagov_serves_the_business_signal_history() -> None:
    today = timezone.localdate()
    run = build_adapter("datagov", _weekdays(today, today)).run("business_signal", today, today)
    periods = MacroIndicator.objects.order_by("period").values_list("period", flat=True)
    logger.info(
        "datagov business_signal: %d rows, %s..%s", run.rows, periods.first(), periods.last()
    )
    assert periods.first() == SIGNAL_START
    assert run.rows > 500
