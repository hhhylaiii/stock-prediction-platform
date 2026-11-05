from datetime import date

import pytest
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.market.models import Market
from tests.api.conftest import add_bars, add_stock, envelope_of

pytestmark = pytest.mark.django_db

OCT_28, OCT_29, OCT_30 = date(2026, 10, 28), date(2026, 10, 29), date(2026, 10, 30)


def test_stock_detail_returns_master_fields_and_latest_quote(client: Client) -> None:
    add_stock()
    add_bars("2330", [OCT_28, OCT_29, OCT_30], [1050.0, 1062.0, 1085.0])

    response = client.get("/api/stocks/2330")

    assert response.status_code == 200
    body = envelope_of(response)
    assert body["success"] is True
    assert body["error"] is None
    assert body["meta"] == {}
    assert body["data"] == {
        "stock_id": "2330",
        "name": "台積電",
        "market": "TWSE",
        "market_label": "上市",
        "industry": "半導體業",
        "list_date": "1994-09-05",
        "delist_date": None,
        "quote": {
            "date": "2026-10-30",
            "close": 1085.0,
            "previous_date": "2026-10-29",
            "previous_close": 1062.0,
            "reference_price": 1062.0,
            "change": 23.0,
            "change_ratio": round(1085 / 1062 - 1, 6),
            "volume": 1_000_002,
            "turnover": 1_085_000_000,
        },
    }


def test_change_on_an_ex_dividend_day_is_against_the_reference_price(client: Client) -> None:
    add_stock()
    add_bars("2330", [OCT_29, OCT_30], [100.0, 96.0], factors=[0.95, 1.0])

    quote = envelope_of(client.get("/api/stocks/2330"))["data"]["quote"]

    assert quote["previous_close"] == 100.0
    assert quote["reference_price"] == 95.0
    assert quote["change"] == 1.0
    assert quote["change_ratio"] == round(96 / 95 - 1, 6)


def test_reference_price_follows_both_factors_before_a_pending_ex_date(client: Client) -> None:
    add_stock()
    add_bars("2330", [OCT_29, OCT_30], [100.0, 100.0], factors=[0.4, 0.5])

    quote = envelope_of(client.get("/api/stocks/2330"))["data"]["quote"]

    assert (quote["reference_price"], quote["change"]) == (80.0, 20.0)


def test_a_day_without_trades_keeps_the_last_traded_quote(client: Client) -> None:
    add_stock()
    add_bars("2330", [OCT_28, OCT_29, OCT_30], [50.0, 51.0, None])

    quote = envelope_of(client.get("/api/stocks/2330"))["data"]["quote"]

    assert (quote["date"], quote["previous_date"]) == ("2026-10-29", "2026-10-28")
    assert quote["close"] == 51.0


def test_a_single_bar_has_no_change(client: Client) -> None:
    add_stock()
    add_bars("2330", [OCT_30], [20.0])

    quote = envelope_of(client.get("/api/stocks/2330"))["data"]["quote"]

    assert quote["close"] == 20.0
    assert quote["previous_date"] is None
    assert quote["previous_close"] is None
    assert quote["reference_price"] is None
    assert quote["change"] is None
    assert quote["change_ratio"] is None


def test_stock_without_prices_has_no_quote(client: Client) -> None:
    add_stock(stock_id="6999", name="新股", market=Market.TPEX, industry="", list_date=None)

    data = envelope_of(client.get("/api/stocks/6999"))["data"]

    assert data["market"] == "TPEx"
    assert data["market_label"] == "上櫃"
    assert data["list_date"] is None
    assert data["quote"] is None


def test_delisted_stock_is_still_served(client: Client) -> None:
    add_stock(stock_id="2498", name="宏達電舊", delist_date=date(2026, 10, 1))
    add_bars("2498", [date(2026, 9, 29), date(2026, 9, 30)], [30.0, 29.0])

    data = envelope_of(client.get("/api/stocks/2498"))["data"]

    assert data["delist_date"] == "2026-10-01"
    assert data["quote"]["date"] == "2026-09-30"


def test_unknown_stock_is_a_not_found_envelope(client: Client) -> None:
    add_stock()

    response = client.get("/api/stocks/9999")

    assert response.status_code == 404
    body = envelope_of(response)
    assert body["success"] is False
    assert body["error"] == {"code": "not_found", "message": "找不到股票 9999。"}


def test_bars_of_other_stocks_are_ignored(client: Client) -> None:
    add_stock()
    add_stock(stock_id="2317", name="鴻海")
    add_bars("2330", [OCT_29], [1000.0])
    add_bars("2317", [OCT_30], [200.0])

    quote = envelope_of(client.get("/api/stocks/2330"))["data"]["quote"]

    assert (quote["date"], quote["close"]) == ("2026-10-29", 1000.0)


def test_stock_detail_query_count(
    client: Client, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    add_stock()
    add_bars("2330", [OCT_28, OCT_29, OCT_30], [1.0, 2.0, 3.0])

    with django_assert_max_num_queries(2):
        client.get("/api/stocks/2330")
