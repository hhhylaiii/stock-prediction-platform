from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
import responses
from django.core.management import call_command

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.adapters.twse import TwseAdapter
from apps.etl.errors import SourceNotReadyError
from apps.etl.models import EtlRun, RawTwse
from apps.etl.ratelimit import RateLimiter
from apps.etl.replay import reprocess
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, SharesOutstanding, StockMaster

TWSE = "https://www.twse.com.tw/rwd/zh"
TPEX = "https://www.tpex.org.tw/www/zh-tw"
DAY = date(2026, 10, 28)
LATER = date(2026, 10, 30)


@pytest.fixture
def twse(october_2026: TradingDays, twse_limiter: RateLimiter) -> TwseAdapter:
    return TwseAdapter(october_2026, twse_limiter)


@pytest.fixture
def tpex(october_2026: TradingDays, twse_limiter: RateLimiter) -> TpexAdapter:
    return TpexAdapter(october_2026, twse_limiter)


def _load_twse_prices(twse: TwseAdapter, load_fixture: Callable[[str], Any]) -> None:
    responses.get(
        f"{TWSE}/afterTrading/MI_INDEX",
        json=load_fixture("twse/daily_price/mi_index_20261028.json"),
    )
    twse.run("daily_price", DAY, DAY)


def _load_twse_shares(
    twse: TwseAdapter, load_fixture: Callable[[str], Any], name: str, day: date
) -> None:
    responses.get(f"{TWSE}/fund/MI_QFIIS", json=load_fixture(f"twse/shares_outstanding/{name}"))
    twse.run("shares_outstanding", day, day)


@responses.activate
@pytest.mark.django_db
def test_prices_are_rebuilt_from_raw_without_any_request(
    twse: TwseAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    _load_twse_prices(twse, load_fixture)
    stored = {row.stock_id: row.close for row in DailyPrice.objects.all()}
    DailyPrice.objects.filter(stock_id="2330").update(close=Decimal("1"))
    DailyPrice.objects.filter(stock_id="1101").delete()
    responses.reset()
    run = reprocess("twse", "daily_price", DAY, DAY, october_2026)
    assert {row.stock_id: row.close for row in DailyPrice.objects.all()} == stored
    assert len(responses.calls) == 0
    assert (run.status, run.checkpoint["kind"], run.checkpoint["last_date"]) == (
        "success",
        "reprocess",
        "2026-10-28",
    )


@responses.activate
@pytest.mark.django_db
def test_a_day_without_stored_raw_stops_the_run_after_the_days_before_it(
    twse: TwseAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    _load_twse_prices(twse, load_fixture)
    DailyPrice.objects.all().delete()
    responses.reset()
    with pytest.raises(SourceNotReadyError, match="no stored raw"):
        reprocess("twse", "daily_price", DAY, date(2026, 10, 29), october_2026)
    run = EtlRun.objects.filter(checkpoint__kind="reprocess").get()
    assert (run.status, run.checkpoint["last_date"]) == ("failed", "2026-10-28")
    assert DailyPrice.objects.filter(date=DAY).exists()


@responses.activate
@pytest.mark.django_db
def test_posted_requests_replay_too(
    tpex: TpexAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(
        f"{TPEX}/afterTrading/otc", json=load_fixture("tpex/daily_price/otc_20261028.json")
    )
    responses.post(
        f"{TPEX}/afterTrading/dailyQuotes",
        json=load_fixture("tpex/daily_price/daily_quotes_20261028.json"),
    )
    tpex.run("daily_price", DAY, DAY)
    stored = DailyPrice.objects.count()
    DailyPrice.objects.all().delete()
    responses.reset()
    reprocess("tpex", "daily_price", DAY, DAY, october_2026)
    assert DailyPrice.objects.count() == stored
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.django_db
def test_share_rows_of_the_day_are_cleared_before_they_are_rebuilt(
    twse: TwseAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code, name=code, market="TWSE") for code in ("0050", "2330", "2368")
    )
    StockMaster.objects.create(stock_id="6488", name="6488", market="TPEx")
    _load_twse_shares(twse, load_fixture, "mi_qfiis_20261028.json", DAY)
    _load_twse_shares(twse, load_fixture, "mi_qfiis_20261030.json", LATER)
    rebuilt = set(SharesOutstanding.objects.values_list("stock_id", "effective_date", "shares"))
    SharesOutstanding.objects.create(stock_id="2330", effective_date=LATER, shares=1)
    SharesOutstanding.objects.create(stock_id="6488", effective_date=LATER, shares=7)
    responses.reset()
    reprocess("twse", "shares_outstanding", LATER, LATER, october_2026)
    after = set(SharesOutstanding.objects.values_list("stock_id", "effective_date", "shares"))
    assert after == rebuilt | {("6488", LATER, 7)}


@pytest.mark.django_db
@pytest.mark.parametrize("source", ["finmind", "yahoo", "tdcc", "datagov"])
def test_sources_without_dated_requests_cannot_be_reprocessed(
    october_2026: TradingDays, source: str
) -> None:
    with pytest.raises(ValueError, match="reprocess"):
        reprocess(source, "daily_price", DAY, DAY, october_2026)


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_reprocess_command_rebuilds_the_range(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    responses.get(
        f"{TWSE}/afterTrading/MI_INDEX",
        json=load_fixture("twse/daily_price/mi_index_20261028.json"),
    )
    call_command("fetch", source="twse", dataset="daily_price", date="2026-10-28")
    DailyPrice.objects.all().delete()
    responses.reset()
    call_command(
        "reprocess", source="twse", dataset="daily_price", start="2026-10-28", end="2026-10-28"
    )
    assert DailyPrice.objects.filter(date=DAY).exists()
    assert RawTwse.objects.count() == 1
