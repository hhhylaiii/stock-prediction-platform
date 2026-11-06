import io
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
import responses
from django.core.management import call_command

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.adapters.twse import TwseAdapter
from apps.etl.exchange_fill import FILL_CHECK, fill_missing
from apps.etl.models import DataQualityLog, EtlRun, RawTwse
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import (
    DailyChip,
    DailyPrice,
    DailyValuation,
    SharesOutstanding,
    StockMaster,
)

BASE = "https://www.twse.com.tw/rwd/zh"
DAY = date(2026, 10, 28)
HOLDINGS = "twse/foreign_holding/mi_qfiis_20261028.json"
VALUATION = "twse/daily_valuation/bwibbu_20261028.json"

pytestmark = pytest.mark.django_db


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TwseAdapter:
    return TwseAdapter(october_2026, twse_limiter)


def priced(*codes: str, day: date = DAY) -> None:
    DailyPrice.objects.bulk_create(
        DailyPrice(stock_id=code, date=day, close=10, volume=1, turnover=10) for code in codes
    )


def serve(load_fixture: Callable[[str], Any], path: str, fixture: str, day: date = DAY) -> None:
    responses.get(f"{BASE}{path}", json={**load_fixture(fixture), "date": f"{day:%Y%m%d}"})


@responses.activate
def test_a_missing_holding_is_filled_and_a_stored_one_left_alone(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS)
    priced("1101", "2317", "2330")
    DailyChip.objects.create(stock_id="2317", date=DAY, fi_holding_pct=50.0)
    DailyChip.objects.create(stock_id="1101", date=DAY, margin_balance=5)
    run = fill_missing(adapter, "holdings", DAY, DAY)
    pct = dict(DailyChip.objects.values_list("stock_id", "fi_holding_pct"))
    assert pct == {"1101": 13.36, "2317": 50.0, "2330": pytest.approx(pct["2330"])}
    assert pct["2330"] is not None
    assert DailyChip.objects.get(stock_id="1101").margin_balance == 5
    assert not DailyChip.objects.filter(stock_id__in=["0050", "1213"]).exists()
    assert (run.status, RawTwse.objects.count()) == ("success", 1)


@responses.activate
def test_shares_are_filled_only_before_the_first_stored_count(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS)
    priced("1101", "2317", "2330")
    SharesOutstanding.objects.create(
        stock_id="2330", effective_date=date(2026, 10, 30), shares=1, reason="unknown"
    )
    SharesOutstanding.objects.create(
        stock_id="1101", effective_date=date(2026, 10, 1), shares=1, reason="unknown"
    )
    fill_missing(adapter, "holdings", DAY, DAY)
    filled = dict(
        SharesOutstanding.objects.filter(effective_date=DAY).values_list("stock_id", "shares")
    )
    assert set(filled) == {"2317", "2330"}
    assert SharesOutstanding.objects.filter(stock_id="1101").count() == 1


@responses.activate
def test_a_count_is_stored_again_only_when_it_changes(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    before = date(2026, 10, 27)
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS, before)
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS)
    priced("2330", day=before)
    priced("2330")
    fill_missing(adapter, "holdings", before, DAY)
    assert list(SharesOutstanding.objects.values_list("effective_date", flat=True)) == [before]


@responses.activate
def test_a_missing_valuation_row_is_added_and_a_stored_row_left_alone(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    serve(load_fixture, "/afterTrading/BWIBBU_d", VALUATION)
    priced("1101", "2317")
    DailyValuation.objects.create(stock_id="2317", date=DAY, pe=1.0, pb=1.0, dividend_yield=0.0)
    fill_missing(adapter, "valuation", DAY, DAY)
    rows = {row.stock_id: row for row in DailyValuation.objects.all()}
    assert set(rows) == {"1101", "2317"}
    assert (rows["2317"].pe, rows["1101"].pe, rows["1101"].pb) == (1.0, None, 0.91)


@responses.activate
def test_the_log_counts_listed_and_delisted_stocks_apart(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    serve(load_fixture, "/afterTrading/BWIBBU_d", VALUATION)
    StockMaster.objects.create(stock_id="1101", name="台泥", market="TWSE")
    StockMaster.objects.create(
        stock_id="1213", name="大飲", market="TWSE", delist_date=date(2026, 11, 1)
    )
    priced("1101", "1213")
    fill_missing(adapter, "valuation", DAY, DAY)
    (log,) = DataQualityLog.objects.filter(check_name=FILL_CHECK)
    assert log.detail["totals"] == {"daily_valuation": {"listed": 1, "delisted": 1}}
    assert log.detail["filled"]["daily_valuation"]["1213"] == ["2026-10-28", "2026-10-28", 1]


@responses.activate
def test_a_resumed_fill_keeps_the_first_run_s_share_windows(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS)
    priced("2330")
    stopped = EtlRun.objects.create(
        source="twse",
        dataset="fill:holdings",
        start_date=date(2026, 10, 27),
        end_date=DAY,
        status="failed",
        checkpoint={"last_date": "2026-10-27", "windows": {"2330": "2026-10-29"}, "filled": {}},
    )
    SharesOutstanding.objects.create(
        stock_id="2330", effective_date=date(2026, 10, 27), shares=5, reason="unknown"
    )
    run = fill_missing(adapter, "holdings", date(2026, 10, 27), DAY, resume=stopped)
    assert run.start_date == DAY
    assert SharesOutstanding.objects.filter(stock_id="2330").count() == 2


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_the_command_runs_a_fill(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    serve(load_fixture, "/afterTrading/BWIBBU_d", VALUATION)
    priced("1101")
    out = io.StringIO()
    call_command(
        "fill_from_exchange",
        "--source",
        "twse",
        "--fill",
        "valuation",
        "--start",
        "2026-10-28",
        "--end",
        "2026-10-28",
        stdout=out,
    )
    assert DailyValuation.objects.filter(stock_id="1101").exists()
    assert "daily_valuation 1" in out.getvalue()


@responses.activate
def test_the_otc_market_fills_holdings_and_shares_from_one_page(
    october_2026: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(
        "https://www.tpex.org.tw/www/zh-tw/insti/qfii",
        json=load_fixture("tpex/foreign_holding/qfii_20261028.json"),
    )
    priced("6488", "1240")
    fill_missing(TpexAdapter(october_2026, twse_limiter), "holdings", DAY, DAY)
    assert set(DailyChip.objects.values_list("stock_id", flat=True)) == {"6488", "1240"}
    shares = dict(SharesOutstanding.objects.values_list("stock_id", "shares"))
    assert shares == {"6488": 528_113_725, "1240": 46_443_992}
    assert len(responses.calls) == 1
