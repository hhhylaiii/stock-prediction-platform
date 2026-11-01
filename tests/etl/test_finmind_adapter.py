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
    DailyPrice,
    DailyValuation,
    DividendEvent,
    Market,
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
    def _mock(dataset: str, name: str, stock: str, span: tuple[str, str], **override: Any) -> None:
        payload = load_fixture(f"finmind/{name}.json")
        query = {"dataset": dataset, "data_id": stock, "start_date": span[0], "end_date": span[1]}
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
    mock: Callable[..., None], span: tuple[str, str] = AUGUST, **result: Any
) -> None:
    policy_end = (date.fromisoformat(span[1]) + timedelta(days=30)).isoformat()
    mock("TaiwanStockDividend", "dividend_2884", "2884", (span[0], policy_end))
    mock("TaiwanStockDividendResult", "dividend_result_2884", "2884", span, **result)


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
