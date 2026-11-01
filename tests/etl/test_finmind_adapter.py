import logging
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError
from responses import matchers

from apps.etl.adapters import ADAPTERS
from apps.etl.adapters.finmind import FinmindAdapter
from apps.etl.models import EtlRun, RawFinmind
from apps.market.adjust import refresh_adj_factor
from apps.market.calendar import TradingDays
from apps.market.models import (
    DailyChip,
    DailyPrice,
    DailyValuation,
    DividendEvent,
    FinancialStatement,
    Market,
    MonthlyRevenue,
    ShareChangeReason,
    SharesOutstanding,
    StockMaster,
)

URL = "https://api.finmindtrade.com/api/v4/data"
AUGUST = ("2013-08-01", "2013-08-31")


WEEKDAYS = TradingDays(
    (
        date.fromordinal(day)
        for day in range(date(2010, 1, 1).toordinal(), date(2026, 12, 31).toordinal() + 1)
        if date.fromordinal(day).weekday() < 5
    ),
    date(2010, 1, 1),
    date(2026, 12, 31),
)


@pytest.fixture(autouse=True)
def quiet(monkeypatch: pytest.MonkeyPatch, db: None) -> None:
    monkeypatch.setattr("apps.etl.adapters.TradingDays.from_db", lambda: WEEKDAYS)
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    monkeypatch.setattr("apps.etl.adapters.finmind.time.sleep", lambda seconds: None)
    for code in ("2330", "2884"):
        StockMaster.objects.create(stock_id=code, name=code, market=Market.TWSE)


@pytest.fixture
def mock(load_fixture: Callable[[str], Any]) -> Callable[..., None]:
    def _mock(
        dataset: str, name: str, stock: str | None, span: tuple[str, str], **override: Any
    ) -> None:
        payload = load_fixture(f"finmind/{name}.json")
        query = {"dataset": dataset, "start_date": span[0], "end_date": span[1]}
        if stock is not None:
            query["data_id"] = stock
        responses.get(
            URL,
            json={**payload, **override},
            match=[matchers.query_param_matcher(query)],
        )

    return _mock


def _backfill(dataset: str, stock: str, span: tuple[str, str]) -> None:
    call_command(
        "backfill",
        "--source",
        "finmind",
        "--dataset",
        dataset,
        "--from",
        span[0],
        "--to",
        span[1],
        "--stock",
        stock,
    )


def _factors(stock: str) -> dict[date, float]:
    rows = DailyPrice.objects.filter(stock_id=stock).values_list("date", "adj_factor")
    return dict(rows)


def _mock_dividends(
    mock: Callable[..., None],
    span: tuple[str, str] = AUGUST,
    stock: str = "2884",
    policy: str = "dividend_2884",
    reductions: Any = None,
    split_data: Any = None,
    **result: Any,
) -> None:
    policy_end = (date.fromisoformat(span[1]) + timedelta(days=30)).isoformat()
    own = stock == "2884"
    mock(
        "TaiwanStockDividend", policy, stock, (span[0], policy_end), **({} if own else {"data": []})
    )
    result = result if own else {"data": [], **result}
    mock("TaiwanStockDividendResult", "dividend_result_2884", stock, span, **result)
    mock(
        "TaiwanStockCapitalReductionReferencePrice",
        "capred_2412",
        stock,
        span,
        data=reductions if reductions is not None else [],
    )
    splits = {} if split_data is None else {"data": split_data}
    mock("TaiwanStockSplitPrice", "split_all", None, span, **splits)


def test_finmind_is_registered() -> None:
    assert ADAPTERS["finmind"] is FinmindAdapter
    assert FinmindAdapter.per_stock


@responses.activate
def test_prices_then_dividends_set_the_adjusted_factor(mock: Callable[..., None]) -> None:
    mock("TaiwanStockPrice", "price_2884_201308", "2884", AUGUST)
    _backfill("daily_price", "2884", AUGUST)
    assert set(_factors("2884").values()) == {1.0}
    _mock_dividends(mock)
    _backfill("dividend_event", "2884", AUGUST)
    factors = _factors("2884")
    assert DividendEvent.objects.count() == 1
    assert factors[date(2013, 8, 15)] == 1.0
    assert factors[date(2013, 8, 14)] / factors[date(2013, 8, 15)] == pytest.approx(17.86 / 19.95)
    assert (
        RawFinmind.objects.filter(dataset__in=["dividend_policy", "dividend_result"]).count() == 2
    )


@responses.activate
def test_dividends_first_give_the_same_factors(mock: Callable[..., None]) -> None:
    _mock_dividends(mock)
    _backfill("dividend_event", "2884", AUGUST)
    mock("TaiwanStockPrice", "price_2884_201308", "2884", AUGUST)
    _backfill("daily_price", "2884", AUGUST)
    factors = _factors("2884")
    assert factors[date(2013, 8, 14)] / factors[date(2013, 8, 15)] == pytest.approx(17.86 / 19.95)
    assert refresh_adj_factor("2884") == 0


@responses.activate
def test_valuations_are_loaded(mock: Callable[..., None]) -> None:
    span = ("2026-10-27", "2026-10-30")
    mock("TaiwanStockPER", "per_2330_202610", "2330", span)
    _backfill("daily_valuation", "2330", span)
    day = DailyValuation.objects.get(stock_id="2330", date=date(2026, 10, 27))
    assert (day.pe, day.pb, day.dividend_yield) == pytest.approx((27.82, 9.68, 0.0092))


@responses.activate
def test_share_counts_skip_a_repeat_of_the_stored_count(
    mock: Callable[..., None], load_fixture: Callable[[str], Any]
) -> None:
    first = ("2013-08-01", "2013-10-31")
    mock("TaiwanStockShareholding", "shareholding_2884_2013q3", "2884", first)
    _backfill("shares_outstanding", "2884", first)
    later = ("2013-08-02", "2013-10-31")
    rows = load_fixture("finmind/shareholding_2884_2013q3.json")["data"][1:]
    mock("TaiwanStockShareholding", "shareholding_2884_2013q3", "2884", later, data=rows)
    _backfill("shares_outstanding", "2884", later)
    stored = SharesOutstanding.objects.order_by("effective_date").values_list(
        "effective_date", "shares"
    )
    assert list(stored) == [(date(2013, 8, 1), 5_010_700_000), (date(2013, 9, 6), 5_524_300_000)]


@responses.activate
def test_an_empty_answer_writes_nothing(mock: Callable[..., None]) -> None:
    mock("TaiwanStockPrice", "price_2884_201308", "2884", AUGUST, data=[])
    _backfill("daily_price", "2884", AUGUST)
    assert DailyPrice.objects.count() == 0


def test_whole_market_days_cannot_be_fetched_from_finmind() -> None:
    with pytest.raises(CommandError, match="one stock at a time"):
        call_command(
            "fetch", "--source", "finmind", "--dataset", "daily_price", "--date", "2026-10-28"
        )


def test_unknown_datasets_are_refused(october_2026: TradingDays) -> None:
    adapter = ADAPTERS["finmind"](
        october_2026, __import__("apps.etl.adapters", fromlist=["x"]).limiter_for("finmind")
    )
    with pytest.raises(ValueError, match="no dataset"):
        adapter.fetch("weekly_holding", date(2026, 10, 1), date(2026, 10, 2), "2330")


@responses.activate
def test_an_ex_date_near_the_range_end_keeps_its_amounts(mock: Callable[..., None]) -> None:
    span = ("2013-08-01", "2013-08-20")
    _mock_dividends(mock, span)
    _backfill("dividend_event", "2884", span)
    event = DividendEvent.objects.get()
    assert (event.ex_date, event.cash_dividend, event.stock_dividend) == (
        date(2013, 8, 15),
        Decimal("0.3"),
        Decimal("1.0"),
    )
    assert event.announce_date is None


@responses.activate
def test_a_rerun_without_the_result_keeps_the_stored_factor(mock: Callable[..., None]) -> None:
    _mock_dividends(mock)
    _backfill("dividend_event", "2884", AUGUST)
    _mock_dividends(mock, data=[])
    _backfill("dividend_event", "2884", AUGUST)
    assert DividendEvent.objects.get().adj_factor == pytest.approx(17.86 / 19.95)


@responses.activate
def test_finmind_does_not_overwrite_rows_from_the_primary_source(
    mock: Callable[..., None],
) -> None:
    DailyPrice.objects.create(
        stock_id="2884", date=date(2013, 8, 14), close=Decimal("99"), volume=1, turnover=99
    )
    mock("TaiwanStockPrice", "price_2884_201308", "2884", AUGUST)
    _backfill("daily_price", "2884", AUGUST)
    assert DailyPrice.objects.get(date=date(2013, 8, 14)).close == Decimal("99")
    assert DailyPrice.objects.count() == 21
    assert EtlRun.objects.get(dataset="daily_price").rows == 20


@responses.activate
def test_nulls_from_finmind_become_blank_values(
    mock: Callable[..., None], load_fixture: Callable[[str], Any]
) -> None:
    span = ("2026-10-27", "2026-10-30")
    rows = load_fixture("finmind/per_2330_202610.json")["data"]
    nulled = [{**rows[0], "PER": None}, *rows[1:]]
    mock("TaiwanStockPER", "per_2330_202610", "2330", span, data=nulled)
    _backfill("daily_valuation", "2330", span)
    assert DailyValuation.objects.get(date=date(2026, 10, 27)).pe is None
    assert DailyValuation.objects.count() == 4


@responses.activate
@pytest.mark.parametrize(
    ("dataset", "finmind_dataset", "name", "model"),
    [
        ("daily_price", "TaiwanStockPrice", "price_2884_201308", DailyPrice),
        ("daily_valuation", "TaiwanStockPER", "per_2330_202610", DailyValuation),
        (
            "shares_outstanding",
            "TaiwanStockShareholding",
            "shareholding_2884_2013q3",
            SharesOutstanding,
        ),
    ],
)
def test_dates_missing_from_the_calendar_reject_the_batch(
    mock: Callable[..., None],
    load_fixture: Callable[[str], Any],
    dataset: str,
    finmind_dataset: str,
    name: str,
    model: type[Any],
) -> None:
    rows = load_fixture(f"finmind/{name}.json")["data"]
    stock = rows[0]["stock_id"]
    odd = [*rows, {**rows[-1], "date": "2013-08-17", "NumberOfSharesIssued": 1}]
    mock(finmind_dataset, name, stock, AUGUST, data=odd)
    with pytest.raises(CommandError, match="2013-08-17 is not a trading day"):
        _backfill(dataset, stock, AUGUST)
    assert model.objects.count() == 0
    assert DailyPrice.objects.count() == 0


@responses.activate
def test_an_earlier_range_loaded_later_leaves_no_repeated_count(
    mock: Callable[..., None], load_fixture: Callable[[str], Any]
) -> None:
    rows = load_fixture("finmind/shareholding_2884_2013q3.json")["data"]
    later = ("2013-08-02", "2013-10-31")
    mock("TaiwanStockShareholding", "shareholding_2884_2013q3", "2884", later, data=rows[1:])
    _backfill("shares_outstanding", "2884", later)
    first = ("2013-08-01", "2013-10-31")
    mock("TaiwanStockShareholding", "shareholding_2884_2013q3", "2884", first)
    _backfill("shares_outstanding", "2884", first)
    stored = SharesOutstanding.objects.order_by("effective_date").values_list(
        "effective_date", "shares"
    )
    assert list(stored) == [(date(2013, 8, 1), 5_010_700_000), (date(2013, 9, 6), 5_524_300_000)]


@responses.activate
def test_an_ex_date_on_a_typhoon_closure_is_kept(
    mock: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = date(2024, 7, 25)
    calendar = TradingDays(
        (day for day in WEEKDAYS.between(WEEKDAYS.start, WEEKDAYS.end) if day != closed),
        WEEKDAYS.start,
        WEEKDAYS.end,
    )
    monkeypatch.setattr("apps.etl.adapters.TradingDays.from_db", lambda: calendar)
    span = ("2024-07-01", "2024-07-31")
    _mock_dividends(mock, span)
    _backfill("dividend_event", "2884", span)
    assert DividendEvent.objects.get().ex_date == closed


@responses.activate
def test_shares_rows_with_a_known_reason_are_never_dropped(mock: Callable[..., None]) -> None:
    SharesOutstanding.objects.create(
        stock_id="2884",
        effective_date=date(2013, 9, 10),
        shares=5_524_300_000,
        reason=ShareChangeReason.CAPITAL_INCREASE,
    )
    span = ("2013-08-01", "2013-10-31")
    mock("TaiwanStockShareholding", "shareholding_2884_2013q3", "2884", span)
    _backfill("shares_outstanding", "2884", span)
    kept = SharesOutstanding.objects.get(effective_date=date(2013, 9, 10))
    assert kept.reason == ShareChangeReason.CAPITAL_INCREASE


@responses.activate
def test_events_without_a_factor_are_logged(
    mock: Callable[..., None], caplog: pytest.LogCaptureFixture
) -> None:
    _mock_dividends(mock, data=[])
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.finmind_datasets"):
        _backfill("dividend_event", "2884", AUGUST)
    assert "2884 2013-08-15" in caplog.text
    assert "no ex-right result" in caplog.text


def _gap(stock: str, before: date, after: date) -> tuple[float, float]:
    prev = DailyPrice.objects.get(stock_id=stock, date=before)
    nxt = DailyPrice.objects.get(stock_id=stock, date=after)
    assert prev.close is not None
    assert nxt.open is not None
    raw = float(nxt.open) / float(prev.close) - 1
    adjusted = float(nxt.open) * nxt.adj_factor / (float(prev.close) * prev.adj_factor) - 1
    return raw, adjusted


@responses.activate
def test_a_capital_reduction_is_flat_after_adjustment(
    mock: Callable[..., None],
    load_fixture: Callable[[str], Any],
) -> None:
    span = ("2010-12-15", "2011-02-15")
    StockMaster.objects.create(stock_id="2412", name="中華電", market=Market.TWSE)
    reductions = load_fixture("finmind/capred_2412.json")["data"]
    _mock_dividends(mock, span, "2412", reductions=reductions, data=[])
    _backfill("dividend_event", "2412", span)
    mock("TaiwanStockPrice", "price_2412_201101", "2412", span)
    _backfill("daily_price", "2412", span)
    raw, adjusted = _gap("2412", date(2011, 1, 6), date(2011, 1, 25))
    assert raw > 0.2
    assert abs(adjusted) < 0.02


@responses.activate
def test_a_split_is_flat_after_adjustment(mock: Callable[..., None]) -> None:
    span = ("2025-06-01", "2025-06-30")
    StockMaster.objects.create(stock_id="0050", name="元大台灣50", market=Market.TWSE)
    _mock_dividends(mock, span, "0050", data=[])
    _backfill("dividend_event", "0050", span)
    mock("TaiwanStockPrice", "price_0050_202506", "0050", span)
    _backfill("daily_price", "0050", span)
    raw, adjusted = _gap("0050", date(2025, 6, 10), date(2025, 6, 18))
    assert raw < -0.7
    assert abs(adjusted) < 0.01


@responses.activate
def test_the_whole_market_split_list_is_fetched_once_per_run(mock: Callable[..., None]) -> None:
    span = AUGUST
    _mock_dividends(mock, span, "2884")
    _mock_dividends(mock, span, "2330", data=[])
    call_command(
        "backfill",
        "--source",
        "finmind",
        "--dataset",
        "dividend_event",
        "--from",
        span[0],
        "--to",
        span[1],
        "--stock",
        "2330",
        "--stock",
        "2884",
    )
    split_calls = [c for c in responses.calls if "TaiwanStockSplitPrice" in str(c.request.url)]
    assert len(split_calls) == 1


@responses.activate
def test_a_split_list_without_stock_ids_is_a_schema_change(
    mock: Callable[..., None], load_fixture: Callable[[str], Any]
) -> None:
    rows = load_fixture("finmind/split_all.json")["data"]
    _mock_dividends(
        mock, split_data=[{k: v for k, v in r.items() if k != "stock_id"} for r in rows]
    )
    with pytest.raises(CommandError, match="stock_id"):
        _backfill("dividend_event", "2884", AUGUST)


OCTOBER = ("2026-10-01", "2026-10-31")


def _mock_chip(mock: Callable[..., None]) -> None:
    mock("TaiwanStockInstitutionalInvestorsBuySell", "inst_2330_202610", "2330", OCTOBER)
    mock("TaiwanStockMarginPurchaseShortSale", "margin_2330_202610", "2330", OCTOBER)
    mock("TaiwanDailyShortSaleBalances", "sbl_2330_202610", "2330", OCTOBER)
    mock("TaiwanStockShareholding", "shareholding_2330_202610", "2330", OCTOBER)


@responses.activate
def test_chip_backfill_writes_one_row_per_trading_day(mock: Callable[..., None]) -> None:
    _mock_chip(mock)
    _backfill("daily_chip", "2330", OCTOBER)
    assert DailyChip.objects.count() == 19
    first = DailyChip.objects.get(stock_id="2330", date=date(2026, 10, 1))
    assert (first.fi_net, first.margin_balance, first.fi_holding_pct) == (
        3_090_730,
        30_658_000,
        69.19,
    )
    assert EtlRun.objects.get(dataset="daily_chip").rows == 19


@responses.activate
def test_chip_backfill_never_overwrites_rows_from_the_primary_source(
    mock: Callable[..., None],
) -> None:
    DailyChip.objects.create(stock_id="2330", date=date(2026, 10, 1), fi_net=1)
    _mock_chip(mock)
    _backfill("daily_chip", "2330", OCTOBER)
    kept = DailyChip.objects.get(stock_id="2330", date=date(2026, 10, 1))
    assert (kept.fi_net, kept.margin_balance) == (1, None)
    assert EtlRun.objects.get(dataset="daily_chip").rows == 18


@responses.activate
def test_shareholding_fetched_for_shares_and_chips_is_stored_once(
    mock: Callable[..., None],
) -> None:
    _mock_chip(mock)
    mock("TaiwanStockShareholding", "shareholding_2330_202610", "2330", OCTOBER)
    _backfill("daily_chip", "2330", OCTOBER)
    _backfill("shares_outstanding", "2330", OCTOBER)
    assert RawFinmind.objects.filter(dataset="shareholding").count() == 1


@responses.activate
def test_chip_dates_missing_from_the_calendar_reject_the_batch(
    mock: Callable[..., None], load_fixture: Callable[[str], Any]
) -> None:
    rows = load_fixture("finmind/shareholding_2884_2013q3.json")["data"]
    for dataset in (
        "TaiwanStockInstitutionalInvestorsBuySell",
        "TaiwanStockMarginPurchaseShortSale",
        "TaiwanDailyShortSaleBalances",
    ):
        mock(dataset, "inst_2330_202610", "2884", AUGUST, data=[])
    odd = [*rows, {**rows[-1], "date": "2013-08-17"}]
    mock("TaiwanStockShareholding", "shareholding_2884_2013q3", "2884", AUGUST, data=odd)
    with pytest.raises(CommandError, match="2013-08-17 is not a trading day"):
        _backfill("daily_chip", "2884", AUGUST)
    assert DailyChip.objects.count() == 0


YEAR_2024 = ("2024-01-01", "2024-12-31")
WITH_LOOKBACK = ("2022-12-31", "2024-12-31")


@responses.activate
def test_revenue_backfill_fetches_a_year_earlier_for_growth(mock: Callable[..., None]) -> None:
    mock("TaiwanStockMonthRevenue", "revenue_2330_2023_2024", "2330", WITH_LOOKBACK)
    _backfill("monthly_revenue", "2330", YEAR_2024)
    january = MonthlyRevenue.objects.get(stock_id="2330", period="2024-01")
    assert january.yoy == pytest.approx(215_785_127_000 / 200_050_544_000 - 1)
    assert january.announce_date == date(2024, 2, 13)
    assert MonthlyRevenue.objects.count() == 12
    assert EtlRun.objects.get(dataset="monthly_revenue").rows == 12


@responses.activate
def test_revenue_backfill_keeps_dates_from_the_primary_source(mock: Callable[..., None]) -> None:
    MonthlyRevenue.objects.create(
        stock_id="2330", period="2024-01", revenue=1, announce_date=date(2024, 2, 6)
    )
    mock("TaiwanStockMonthRevenue", "revenue_2330_2023_2024", "2330", WITH_LOOKBACK)
    _backfill("monthly_revenue", "2330", YEAR_2024)
    kept = MonthlyRevenue.objects.get(stock_id="2330", period="2024-01")
    assert (kept.revenue, kept.announce_date) == (1, date(2024, 2, 6))


@responses.activate
def test_statement_backfill_writes_one_row_per_quarter(mock: Callable[..., None]) -> None:
    mock("TaiwanStockFinancialStatements", "income_2330_2023_2024", "2330", WITH_LOOKBACK)
    mock("TaiwanStockBalanceSheet", "balance_2330_2023_2024", "2330", WITH_LOOKBACK)
    mock("TaiwanStockCashFlowsStatement", "cashflow_2330_2023_2024", "2330", WITH_LOOKBACK)
    _backfill("financial_statement", "2330", YEAR_2024)
    periods = FinancialStatement.objects.order_by("period").values_list("period", flat=True)
    assert list(periods) == ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"]
    q2 = FinancialStatement.objects.get(period="2024-Q2")
    assert (q2.fin_type, q2.operating_cf) == ("consolidated", 813_979_318_000 - 436_311_108_000)
    assert q2.announce_date == date(2024, 9, 3)
    assert RawFinmind.objects.filter(dataset="balance_sheet").count() == 1


@responses.activate
def test_one_statement_losing_a_key_fails_the_batch(
    mock: Callable[..., None], load_fixture: Callable[[str], Any]
) -> None:
    balance = load_fixture("finmind/balance_2330_2023_2024.json")["data"]
    mock("TaiwanStockFinancialStatements", "income_2330_2023_2024", "2330", WITH_LOOKBACK)
    mock(
        "TaiwanStockBalanceSheet",
        "balance_2330_2023_2024",
        "2330",
        WITH_LOOKBACK,
        data=[{k: v for k, v in row.items() if k != "value"} for row in balance],
    )
    mock("TaiwanStockCashFlowsStatement", "cashflow_2330_2023_2024", "2330", WITH_LOOKBACK)
    with pytest.raises(CommandError, match="value"):
        _backfill("financial_statement", "2330", YEAR_2024)
    assert FinancialStatement.objects.count() == 0


@responses.activate
def test_prices_before_the_listing_date_are_not_written(mock: Callable[..., None]) -> None:
    StockMaster.objects.filter(stock_id="2884").update(list_date=date(2013, 8, 15))
    mock("TaiwanStockPrice", "price_2884_201308", "2884", AUGUST)
    _backfill("daily_price", "2884", AUGUST)
    assert min(DailyPrice.objects.values_list("date", flat=True)) == date(2013, 8, 15)


@responses.activate
def test_chips_before_the_listing_date_are_not_written(mock: Callable[..., None]) -> None:
    StockMaster.objects.filter(stock_id="2330").update(list_date=date(2026, 10, 15))
    _mock_chip(mock)
    _backfill("daily_chip", "2330", OCTOBER)
    assert min(DailyChip.objects.values_list("date", flat=True)) == date(2026, 10, 15)
