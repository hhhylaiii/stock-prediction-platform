from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError
from responses import matchers

from apps.etl.adapters import ADAPTERS
from apps.etl.adapters.finmind import FinmindAdapter
from apps.etl.models import RawFinmind
from apps.market.adjust import refresh_adj_factor
from apps.market.calendar import TradingDays
from apps.market.models import (
    DailyPrice,
    DailyValuation,
    DividendEvent,
    Market,
    SharesOutstanding,
    StockMaster,
)

URL = "https://api.finmindtrade.com/api/v4/data"
AUGUST = ("2013-08-01", "2013-08-31")


@pytest.fixture(autouse=True)
def quiet(monkeypatch: pytest.MonkeyPatch, october_2026_in_db: TradingDays) -> None:
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


def _mock_dividends(mock: Callable[..., None]) -> None:
    mock("TaiwanStockDividend", "dividend_2884", "2884", AUGUST)
    mock("TaiwanStockDividendResult", "dividend_result_2884", "2884", AUGUST)


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
    later = DividendEvent.objects.filter(ex_date__gt=date(2013, 8, 15))
    expected_after = 1.0
    for event in later:
        assert event.adj_factor is not None
        expected_after *= event.adj_factor
    assert DividendEvent.objects.count() == 21
    assert factors[date(2013, 8, 15)] == pytest.approx(expected_after)
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
