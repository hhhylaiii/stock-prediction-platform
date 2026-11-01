import logging
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import responses

from apps.etl.adapters.twse import TwseAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import RawTwse
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import DailyValuation, SharesOutstanding

VALUATION_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
HOLDING_URL = "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"
VALUATION = "twse/daily_valuation/bwibbu_20261028.json"
VALUATION_NO_DATA = "twse/daily_valuation/bwibbu_20261101_no_data.json"
SHARES = {
    date(2026, 10, 28): "twse/shares_outstanding/mi_qfiis_20261028.json",
    date(2026, 10, 30): "twse/shares_outstanding/mi_qfiis_20261030.json",
}
DAY = date(2026, 10, 28)
NEXT = date(2026, 10, 30)


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TwseAdapter:
    return TwseAdapter(october_2026, twse_limiter)


def _rows(frame: Any, key: str = "stock_id") -> dict[str, dict[str, Any]]:
    return {
        str(row[key]): {str(name): value for name, value in row.items()}
        for row in frame.to_dict("records")
    }


def _shares_in_db() -> dict[tuple[str, date], int]:
    return {
        (row.stock_id, row.effective_date): row.shares for row in SharesOutstanding.objects.all()
    }


def _load_shares(adapter: TwseAdapter, payload: dict[str, Any], day: date) -> int:
    responses.get(HOLDING_URL, json=payload)
    frame = adapter.normalize("shares_outstanding", adapter.fetch("shares_outstanding", day, day))
    return adapter.upsert("shares_outstanding", frame)


@responses.activate
@pytest.mark.django_db
def test_valuation_sends_the_day_and_stores_raw(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(VALUATION)
    responses.get(VALUATION_URL, json=payload)
    adapter.fetch("daily_valuation", DAY, DAY)
    stored = RawTwse.objects.get()
    assert (stored.dataset, stored.payload) == ("daily_valuation", payload)
    assert stored.request_params == {"date": "20261028", "selectType": "ALL", "response": "json"}
    sent = parse_qs(urlparse(responses.calls[0].request.url or "").query)
    assert {key: values[0] for key, values in sent.items()} == stored.request_params


@responses.activate
@pytest.mark.django_db
def test_valuation_keeps_ratios_and_turns_yield_percent_into_a_fraction(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(VALUATION_URL, json=load_fixture(VALUATION))
    rows = _rows(adapter.normalize("daily_valuation", adapter.fetch("daily_valuation", DAY, DAY)))
    assert rows["2330"] == {
        "stock_id": "2330",
        "date": DAY,
        "pe": 28.11,
        "pb": 9.78,
        "dividend_yield": pytest.approx(0.0091),
    }
    assert (rows["1101"]["pe"], rows["1101"]["pb"]) == (None, 0.91)
    assert rows["1213"]["dividend_yield"] == 0.0


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("column", "value", "dropped"),
    [(5, "0.00", "pe"), (6, "-1.20", "pb"), (3, "100.00", "dividend_yield")],
)
def test_valuation_out_of_range_is_dropped_with_a_warning(
    adapter: TwseAdapter,
    load_fixture: Callable[[str], Any],
    caplog: pytest.LogCaptureFixture,
    column: int,
    value: str,
    dropped: str,
) -> None:
    payload = load_fixture(VALUATION)
    next(row for row in payload["data"] if row[0] == "2330")[column] = value
    responses.get(VALUATION_URL, json=payload)
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.twse_tables"):
        frame = adapter.normalize("daily_valuation", adapter.fetch("daily_valuation", DAY, DAY))
    assert _rows(frame)["2330"][dropped] is None
    assert "2330" in caplog.text


@responses.activate
@pytest.mark.django_db
def test_valuation_word_in_a_ratio_column_is_a_schema_change(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(VALUATION)
    next(row for row in payload["data"] if row[0] == "2330")[5] = "N/A"
    responses.get(VALUATION_URL, json=payload)
    with pytest.raises(SchemaChangedError, match="N/A"):
        adapter.normalize("daily_valuation", adapter.fetch("daily_valuation", DAY, DAY))


@responses.activate
@pytest.mark.django_db
def test_valuation_without_percent_column_is_a_schema_change(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(VALUATION)
    payload["fields"][3] = "殖利率"
    responses.get(VALUATION_URL, json=payload)
    with pytest.raises(SchemaChangedError, match="殖利率"):
        adapter.fetch("daily_valuation", DAY, DAY)
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_valuation_unpublished_day_is_not_ready(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(VALUATION_URL, json=load_fixture(VALUATION_NO_DATA))
    with pytest.raises(SourceNotReadyError):
        adapter.fetch("daily_valuation", DAY, DAY)
    assert RawTwse.objects.count() == 0


@responses.activate
@pytest.mark.django_db
def test_valuation_upsert_is_idempotent(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(VALUATION)
    for _ in range(2):
        responses.get(VALUATION_URL, json=payload)
        frame = adapter.normalize("daily_valuation", adapter.fetch("daily_valuation", DAY, DAY))
        assert adapter.upsert("daily_valuation", frame) == 4
    tsmc = DailyValuation.objects.get(stock_id="2330", date=DAY)
    assert (tsmc.pe, tsmc.pb) == (28.11, 9.78)
    assert DailyValuation.objects.count() == 4
