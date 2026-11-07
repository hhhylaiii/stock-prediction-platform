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
from apps.etl.errors import SourceNotReadyError
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
MARGIN = "twse/margin/mi_margn_20261028.json"
LENDING = "twse/sbl/twt93u_20261028.json"
OLD_LENDING = "twse/sbl/twt93u_20100104_old_layout.json"
MARGINS = ("margin_balance", "margin_limit", "short_balance")

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


def margins(code: str) -> tuple[Any, ...]:
    return tuple(DailyChip.objects.values_list(*MARGINS).get(stock_id=code))


@responses.activate
def test_missing_margin_rows_are_added_and_only_blank_margin_columns_filled(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    serve(load_fixture, "/marginTrading/MI_MARGN", MARGIN)
    priced("1101", "1213", "2317", "2330", "2603", "2609")
    DailyChip.objects.create(stock_id="1101", date=DAY, fi_holding_pct=13.36)
    stored = dict(zip(MARGINS, (35_332_000, 6_483_092_000, 5_000), strict=True))
    DailyChip.objects.create(stock_id="2330", date=DAY, **stored)
    DailyChip.objects.create(stock_id="2317", date=DAY, margin_balance=5)
    DailyChip.objects.create(stock_id="2609", date=DAY, fi_net=1)
    run = fill_missing(adapter, "margin", DAY, DAY)
    assert margins("1101") == (37_423_000, 1_880_795_000, 501_000)
    assert DailyChip.objects.get(stock_id="1101").fi_holding_pct == 13.36
    assert margins("2317") == (5, 3_518_936_000, 471_000)
    assert margins("1213") == (0, 0, 0)
    assert margins("2609") == (None, None, None)
    assert margins("2330") == tuple(stored.values())
    assert not DailyChip.objects.filter(stock_id__in=["0050", "2603"]).exists()
    (log,) = DataQualityLog.objects.filter(check_name=FILL_CHECK)
    assert sorted(log.detail["filled"]["margin"]) == ["1101", "1213", "2317"]
    assert log.detail["created"]["margin"] == {"1213": ["2026-10-28", "2026-10-28", 1]}
    assert log.detail["differences"]["margin"]["sample"] == [
        ["2317", "2026-10-28", "margin_balance", 5, 56_619_000]
    ]
    assert (run.status, run.rows) == ("success", 3)


@responses.activate
def test_the_otc_market_fills_margins_from_its_balance_page(
    october_2026: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(
        "https://www.tpex.org.tw/www/zh-tw/margin/balance",
        json=load_fixture("tpex/margin/balance_20261028.json"),
    )
    priced("1240", "3293")
    fill_missing(TpexAdapter(october_2026, twse_limiter), "margin", DAY, DAY)
    assert margins("1240") == (14_000, 11_610_000, 0)
    assert margins("3293")[0] == 1_976_000
    assert len(responses.calls) == 1


def lent(code: str) -> int | None:
    return DailyChip.objects.values_list("sbl_short_balance", flat=True).get(stock_id=code)


@responses.activate
def test_missing_lending_balances_are_added_and_stored_ones_left_alone(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    serve(load_fixture, "/marginTrading/TWT93U", LENDING)
    priced("1101", "1213", "2317", "2330", "2603", "2609")
    DailyChip.objects.create(stock_id="1101", date=DAY, fi_holding_pct=13.36)
    DailyChip.objects.create(stock_id="2317", date=DAY, sbl_short_balance=5)
    DailyChip.objects.create(stock_id="2330", date=DAY, sbl_short_balance=15_528_513)
    DailyChip.objects.create(stock_id="2609", date=DAY, fi_net=1)
    run = fill_missing(adapter, "sbl", DAY, DAY)
    assert [lent(code) for code in ("1101", "1213", "2317", "2330", "2609")] == [
        113_290_742,
        0,
        5,
        15_528_513,
        None,
    ]
    assert DailyChip.objects.get(stock_id="1101").fi_holding_pct == 13.36
    assert not DailyChip.objects.filter(stock_id__in=["0050", "2603"]).exists()
    (log,) = DataQualityLog.objects.filter(check_name=FILL_CHECK)
    assert sorted(log.detail["filled"]["sbl"]) == ["1101", "1213"]
    assert log.detail["created"]["sbl"] == {"1213": ["2026-10-28", "2026-10-28", 1]}
    assert log.detail["differences"]["sbl"]["sample"] == [
        ["2317", "2026-10-28", "sbl_short_balance", 5, 77_830_749],
        ["2330", "2026-10-28", "sbl_short_balance", 15_528_513, 15_528_514],
    ]
    assert (run.status, run.rows) == ("success", 2)


@responses.activate
def test_lending_pages_before_the_march_2012_formula_fill_from_their_old_balance(
    january_2010: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    day = date(2010, 1, 4)
    serve(load_fixture, "/marginTrading/TWT93U", OLD_LENDING, day)
    priced("1101", "2330", day=day)
    DailyChip.objects.create(stock_id="2330", date=day, sbl_short_balance=41_036_580)
    run = fill_missing(TwseAdapter(january_2010, twse_limiter), "sbl", day, day)
    assert [lent(code) for code in ("1101", "2330")] == [1_035_500, 41_036_580]
    (log,) = DataQualityLog.objects.filter(check_name=FILL_CHECK)
    assert log.detail["differences"]["sbl"]["count"] == 0
    assert (run.status, run.rows) == ("success", 1)


@responses.activate
def test_the_otc_market_fills_lending_balances_from_its_sbl_page(
    october_2026: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(
        "https://www.tpex.org.tw/www/zh-tw/margin/sbl",
        json=load_fixture("tpex/sbl/sbl_20261028.json"),
    )
    priced("1240", "3293")
    fill_missing(TpexAdapter(october_2026, twse_limiter), "sbl", DAY, DAY)
    assert (lent("1240"), lent("3293")) == (17_310, 4_159_575)
    assert len(responses.calls) == 1


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
            "listed": 1,
            "delisted": 0,
        },
        "shares_outstanding": {
            "count": 1,
            "sample": [["2330", "2026-10-28", "shares", 5, 25_932_370_067]],
            "listed": 1,
            "delisted": 0,
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
        "listed": 1,
        "delisted": 0,
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


def holdings_on(load_fixture: Callable[[str], Any], day: date, shares: str) -> None:
    page = edited(load_fixture, HOLDINGS, day, {"2330": tsmc_with(load_fixture, shares)})
    responses.get(f"{BASE}/fund/MI_QFIIS", json=page)


@responses.activate
@pytest.mark.parametrize(
    ("bound", "dates", "differing", "dropped"),
    [
        (date(2026, 10, 30), ["2026-10-27", "2026-10-28", "2026-10-30"], 0, []),
        (date(2026, 10, 28), ["2026-10-27"], 1, [["2330", "2026-10-28", 25_932_370_067]]),
    ],
)
def test_a_stored_first_count_stays_until_the_fill_has_passed_it(
    adapter: TwseAdapter,
    load_fixture: Callable[[str], Any],
    bound: date,
    dates: list[str],
    differing: int,
    dropped: list[list[Any]],
) -> None:
    before = date(2026, 10, 27)
    holdings_on(load_fixture, before, "25,932,370,067")
    holdings_on(load_fixture, DAY, "26,000,000,000")
    priced("2330", day=before)
    priced("2330")
    SharesOutstanding.objects.create(
        stock_id="2330", effective_date=bound, shares=25_932_370_067, reason="unknown"
    )
    fill_missing(adapter, "holdings", before, DAY)
    kept = SharesOutstanding.objects.order_by("effective_date").values_list(
        "effective_date", flat=True
    )
    assert [str(day) for day in kept] == dates
    (log,) = DataQualityLog.objects.filter(check_name=FILL_CHECK)
    assert log.detail["differences"].get("shares_outstanding", {}).get("count", 0) == differing
    assert log.detail["dropped"] == dropped


@responses.activate
def test_a_day_that_fails_part_way_writes_nothing_and_keeps_its_checkpoint(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    before = date(2026, 10, 27)
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS, before)
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS)
    priced("2330", day=before)
    priced("2330")
    fill = exchange_fill.Shares.fill

    def failing(self: exchange_fill.Shares, frame: Any, day: date) -> exchange_fill.Written:
        if day == DAY:
            raise RuntimeError("stopped")
        return fill(self, frame, day)

    monkeypatch.setattr(exchange_fill.Shares, "fill", failing)
    with pytest.raises(RuntimeError):
        fill_missing(adapter, "holdings", before, DAY)
    assert not DailyChip.objects.filter(date=DAY).exists()
    assert EtlRun.objects.get().checkpoint["last_date"] == "2026-10-27"


@responses.activate
@pytest.mark.parametrize(("stored", "differs"), [(1.84, True), (1.8504, False)])
def test_a_value_one_published_digit_off_is_a_difference(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any], stored: float, differs: bool
) -> None:
    serve(load_fixture, "/afterTrading/BWIBBU_d", VALUATION)
    priced("2317")
    DailyValuation.objects.create(
        stock_id="2317", date=DAY, pe=16.58, pb=stored, dividend_yield=0.0285
    )
    fill_missing(adapter, "valuation", DAY, DAY)
    (log,) = DataQualityLog.objects.filter(check_name=FILL_CHECK)
    assert bool(log.detail["differences"]["daily_valuation"]["count"]) == differs


@responses.activate
def test_a_table_none_of_whose_codes_is_priced_waits(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    serve(load_fixture, "/fund/MI_QFIIS", HOLDINGS)
    with pytest.raises(SourceNotReadyError, match="none of the 5 listed codes is priced"):
        fill_missing(adapter, "holdings", DAY, DAY)


@pytest.mark.usefixtures("october_2026_in_db")
def test_a_start_after_the_end_is_refused() -> None:
    with pytest.raises(CommandError, match="after"):
        call_command(
            "fill_from_exchange",
            "--source",
            "twse",
            "--fill",
            "valuation",
            "--start",
            "2026-10-28",
            "--end",
            "2026-10-27",
        )
