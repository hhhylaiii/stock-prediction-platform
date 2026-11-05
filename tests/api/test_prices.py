from datetime import date

import pytest
from django.db import connection
from django.test import Client
from django.test.utils import CaptureQueriesContext
from pytest_django import DjangoAssertNumQueries

from apps.market.models import TradingCalendar
from tests.api.conftest import add_bars, add_stock, envelope_of, sessions_until

pytestmark = pytest.mark.usefixtures("calendar_in_db")

OCT_30 = date(2026, 10, 30)
LAST_70 = sessions_until(OCT_30)[-70:]


def test_default_series_is_the_last_60_sessions_oldest_first(client: Client) -> None:
    add_stock()
    add_bars("2330", LAST_70, [100.0 + n for n in range(70)])

    response = client.get("/api/stocks/2330/prices")

    assert response.status_code == 200
    body = envelope_of(response)
    bars = body["data"]
    assert [bar["date"] for bar in bars] == [day.isoformat() for day in LAST_70[-60:]]
    assert "2026-10-09" not in {bar["date"] for bar in bars}
    assert bars[-1] == {
        "date": "2026-10-30",
        "open": 168.0,
        "high": 171.0,
        "low": 167.0,
        "close": 169.0,
        "volume": 1_000_069,
        "turnover": 169_000_000,
    }
    assert body["meta"] == {
        "days": 60,
        "start_date": LAST_70[-60].isoformat(),
        "end_date": "2026-10-30",
        "adjusted_to": "2026-10-30",
    }


def test_prices_before_an_ex_date_are_adjusted(client: Client) -> None:
    days = LAST_70[-3:]
    add_stock()
    add_bars("2330", days, [100.0, 100.0, 90.0], factors=[0.9, 0.9, 1.0])

    bars = envelope_of(client.get("/api/stocks/2330/prices", {"days": 3}))["data"]

    assert [bar["close"] for bar in bars] == [90.0, 90.0, 90.0]
    assert [bar["open"] for bar in bars] == [89.1, 89.1, 89.0]
    assert [bar["volume"] for bar in bars] == [1_000_000, 1_000_001, 1_000_002]


def test_series_ends_on_the_latest_raw_price_even_with_a_pending_factor(client: Client) -> None:
    days = LAST_70[-2:]
    add_stock()
    add_bars("2330", days, [100.0, 100.0], factors=[0.4, 0.5])

    bars = envelope_of(client.get("/api/stocks/2330/prices", {"days": 2}))["data"]

    assert [bar["close"] for bar in bars] == [80.0, 100.0]


def test_missing_and_untraded_sessions_stay_as_empty_bars(client: Client) -> None:
    days = LAST_70[-4:]
    add_stock()
    add_bars("2330", [days[0], days[2], days[3]], [10.0, None, 12.0])

    bars = envelope_of(client.get("/api/stocks/2330/prices", {"days": 4}))["data"]

    assert [bar["date"] for bar in bars] == [day.isoformat() for day in days]
    assert bars[1] == {
        "date": days[1].isoformat(),
        "open": None,
        "high": None,
        "low": None,
        "close": None,
        "volume": None,
        "turnover": None,
    }
    assert (bars[2]["close"], bars[2]["volume"], bars[2]["turnover"]) == (None, 0, 0)


def test_sessions_missing_at_the_window_start_stay_as_empty_bars(client: Client) -> None:
    days = LAST_70[-8:]
    add_stock()
    add_bars("2330", [*days[:2], *days[5:]], [10.0] * 5)

    body = envelope_of(client.get("/api/stocks/2330/prices", {"days": 6}))

    assert [bar["date"] for bar in body["data"]] == [day.isoformat() for day in days[2:]]
    assert [bar["close"] for bar in body["data"]] == [None, None, None, 10.0, 10.0, 10.0]
    assert body["meta"]["start_date"] == days[2].isoformat()


def test_series_of_a_new_listing_starts_at_its_first_bar(client: Client) -> None:
    add_stock()
    add_bars("2330", LAST_70[-5:], [10.0] * 5)

    body = envelope_of(client.get("/api/stocks/2330/prices"))

    assert len(body["data"]) == 5
    assert body["meta"]["start_date"] == LAST_70[-5].isoformat()


def test_delisted_series_ends_at_its_last_bar(client: Client) -> None:
    last = LAST_70[-20]
    add_stock(delist_date=LAST_70[-19])
    add_bars("2330", LAST_70[-30:-19], [5.0] * 11)

    body = envelope_of(client.get("/api/stocks/2330/prices", {"days": 5}))

    assert body["data"][-1]["date"] == last.isoformat()
    assert body["meta"]["end_date"] == last.isoformat()


def test_stock_without_prices_has_an_empty_series(client: Client) -> None:
    add_stock()

    body = envelope_of(client.get("/api/stocks/2330/prices"))

    assert body["data"] == []
    assert body["meta"] == {"days": 60, "start_date": None, "end_date": None, "adjusted_to": None}


@pytest.mark.parametrize("days", ["0", "501", "abc", "-3"])
def test_days_outside_one_to_500_is_invalid(client: Client, days: str) -> None:
    add_stock()

    response = client.get("/api/stocks/2330/prices", {"days": days})

    assert response.status_code == 400
    assert set(envelope_of(response)["error"]["fields"]) == {"days"}


def test_unknown_stock_prices_is_not_found(client: Client) -> None:
    response = client.get("/api/stocks/9999/prices")

    assert response.status_code == 404
    assert envelope_of(response)["error"] == {"code": "not_found", "message": "找不到股票 9999。"}


def test_prices_past_the_calendar_report_an_outdated_calendar(client: Client) -> None:
    add_stock()
    add_bars("2330", [date(2026, 12, 1)], [10.0])

    response = client.get("/api/stocks/2330/prices")

    assert response.status_code == 503
    assert envelope_of(response)["error"]["code"] == "calendar_outdated"


def test_empty_calendar_reports_it_unavailable(client: Client) -> None:
    add_stock()
    add_bars("2330", [OCT_30], [10.0])
    TradingCalendar.objects.all().delete()

    response = client.get("/api/stocks/2330/prices")

    assert response.status_code == 503
    assert envelope_of(response)["error"]["code"] == "calendar_unavailable"


def test_only_bars_inside_the_window_are_loaded(client: Client) -> None:
    add_stock()
    add_bars("2330", LAST_70, [1.0] * 70)

    with CaptureQueriesContext(connection) as queries:
        client.get("/api/stocks/2330/prices", {"days": 3})

    bounded = f"BETWEEN '{LAST_70[-3].isoformat()}'::date AND '2026-10-30'::date"
    loaded = [query["sql"] for query in queries if "BETWEEN" in query["sql"]]
    assert len(loaded) == 1
    assert '"daily_price"' in loaded[0]
    assert bounded in loaded[0]


def test_prices_query_count(
    client: Client, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    add_stock()
    add_bars("2330", LAST_70, [1.0] * 70)

    with django_assert_max_num_queries(4):
        client.get("/api/stocks/2330/prices")


def test_prices_before_the_calendar_report_it_unavailable(client: Client) -> None:
    add_stock()
    add_bars("2330", [date(2026, 6, 30)], [10.0])

    response = client.get("/api/stocks/2330/prices")

    assert response.status_code == 503
    assert envelope_of(response)["error"]["code"] == "calendar_unavailable"
