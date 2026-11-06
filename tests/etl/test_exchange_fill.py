import io
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl import exchange_fill
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


def edited(
    load_fixture: Callable[[str], Any], fixture: str, day: date, rows: dict[str, list[Any]]
) -> dict[str, Any]:
    page = load_fixture(fixture)
    kept = [rows.get(row[0], row) for row in page["data"]]
    extra = [row for code, row in rows.items() if code not in {row[0] for row in page["data"]}]
    data = kept + extra
    return {**page, "date": f"{day:%Y%m%d}", "data": data, "total": len(data)}


def tsmc_with(load_fixture: Callable[[str], Any], shares: str) -> list[Any]:
    (row,) = [row for row in load_fixture(HOLDINGS)["data"] if row[0] == "2330"]
    return [*row[:3], shares, *row[4:]]


@responses.activate
def test_stored_values_that_differ_from_the_exchange_are_logged(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS)
    priced("1101", "2317", "2330")
    DailyChip.objects.create(stock_id="2317", date=DAY, fi_holding_pct=50.0)
    DailyChip.objects.create(stock_id="1101", date=DAY, fi_holding_pct=13.36)
    SharesOutstanding.objects.create(
        stock_id="2330", effective_date=DAY, shares=5, reason="unknown"
    )
    fill_missing(adapter, "holdings", DAY, DAY)
    (log,) = DataQualityLog.objects.filter(check_name=FILL_CHECK)
    assert log.detail["differences"] == {
        "foreign_holding": {
            "count": 1,
            "sample": [["2317", "2026-10-28", "fi_holding_pct", 50.0, 39.96]],
        },
        "shares_outstanding": {
            "count": 1,
            "sample": [["2330", "2026-10-28", "shares", 5, 25_932_370_067]],
        },
    }
    assert list(
        SharesOutstanding.objects.filter(stock_id="2330").values_list("shares", flat=True)
    ) == [5]


@responses.activate
def test_stored_valuations_that_differ_are_logged_and_blank_ones_not_added(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    blank = ["9999", "空白", "10.00", "-", 114, "-", "-", "115/2"]
    page = edited(load_fixture, VALUATION, DAY, {"9999": blank})
    responses.get(f"{BASE}/afterTrading/BWIBBU_d", json=page)
    priced("2317", "9999")
    DailyValuation.objects.create(
        stock_id="2317", date=DAY, pe=16.58, pb=1.0, dividend_yield=0.0285
    )
    fill_missing(adapter, "valuation", DAY, DAY)
    assert not DailyValuation.objects.filter(stock_id="9999").exists()
    (log,) = DataQualityLog.objects.filter(check_name=FILL_CHECK)
    assert log.detail["differences"]["daily_valuation"] == {
        "count": 1,
        "sample": [["2317", "2026-10-28", "pb", 1.0, 1.85]],
    }


@responses.activate
def test_a_count_repeating_the_first_stored_one_is_not_kept_twice(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS)
    priced("2330")
    SharesOutstanding.objects.create(
        stock_id="2330", effective_date=date(2026, 10, 30), shares=25_932_370_067, reason="unknown"
    )
    fill_missing(adapter, "holdings", DAY, DAY)
    assert list(SharesOutstanding.objects.values_list("effective_date", flat=True)) == [DAY]


@responses.activate
def test_a_second_fill_keeps_the_bounds_the_first_one_saw(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    before = date(2026, 10, 27)
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS, before)
    changed = edited(
        load_fixture, HOLDINGS, DAY, {"2330": tsmc_with(load_fixture, "26,000,000,000")}
    )
    responses.get(f"{BASE}/fund/MI_QFIIS", json=changed)
    priced("2330", day=before)
    priced("2330")
    fill_missing(adapter, "holdings", before, before)
    fill_missing(adapter, "holdings", DAY, DAY)
    counts = list(
        SharesOutstanding.objects.order_by("effective_date").values_list("shares", flat=True)
    )
    assert counts == [25_932_370_067, 26_000_000_000]


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_a_stopped_fill_goes_on_from_its_checkpoint(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    before = date(2026, 10, 27)
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS, before)
    changed = edited(
        load_fixture, HOLDINGS, DAY, {"2330": tsmc_with(load_fixture, "26,000,000,000")}
    )
    responses.get(f"{BASE}/fund/MI_QFIIS", json=changed)
    priced("2330", day=before)
    priced("2330")
    prices = exchange_fill._priced
    stops = iter([True, False])

    def stopping(day: date) -> set[str]:
        if day == DAY and next(stops):
            raise RuntimeError("stopped")
        return prices(day)

    monkeypatch.setattr(exchange_fill, "_priced", stopping)
    args = ["fill_from_exchange", "--source", "twse", "--fill", "holdings"]
    with pytest.raises(RuntimeError):
        call_command(*args, "--start", "2026-10-27", "--end", "2026-10-28", stdout=io.StringIO())
    stopped = EtlRun.objects.get(dataset="fill:holdings")
    assert (stopped.status, stopped.checkpoint["last_date"]) == ("failed", "2026-10-27")
    assert stopped.checkpoint["windows"] == {}
    call_command(*args, "--resume", str(stopped.run_id), stdout=io.StringIO())
    counts = list(
        SharesOutstanding.objects.order_by("effective_date").values_list("shares", flat=True)
    )
    assert counts == [25_932_370_067, 26_000_000_000]
    (log,) = DataQualityLog.objects.filter(check_name=FILL_CHECK)
    assert log.detail["span"] == ["2026-10-27", "2026-10-28"]
    assert log.detail["filled"]["shares_outstanding"]["2330"] == ["2026-10-27", "2026-10-28", 2]


def test_a_fill_stopped_after_its_last_day_only_writes_its_log(adapter: TwseAdapter) -> None:
    stopped = EtlRun.objects.create(
        source="twse",
        dataset="fill:valuation",
        start_date=DAY,
        end_date=DAY,
        status="failed",
        checkpoint={"start": "2026-10-27", "last_date": "2026-10-28", "filled": {}},
    )
    fill_missing(adapter, "valuation", DAY, DAY, resume=stopped)
    (log,) = DataQualityLog.objects.filter(check_name=FILL_CHECK)
    assert log.detail["span"] == ["2026-10-27", "2026-10-28"]


@pytest.mark.usefixtures("october_2026_in_db")
def test_a_fill_still_marked_running_is_not_resumed() -> None:
    running = EtlRun.objects.create(
        source="twse", dataset="fill:holdings", start_date=DAY, end_date=DAY, checkpoint={}
    )
    with pytest.raises(CommandError, match="still marked running"):
        call_command(
            "fill_from_exchange",
            "--source",
            "twse",
            "--fill",
            "holdings",
            "--resume",
            str(running.run_id),
        )
