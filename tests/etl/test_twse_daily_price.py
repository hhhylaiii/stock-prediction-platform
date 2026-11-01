import copy
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import requests
import responses
from django.core.exceptions import ImproperlyConfigured

from apps.etl.adapters.twse import TwseAdapter, is_foreign_currency_etf, quotes_frame
from apps.etl.errors import SchemaChangedError, SourceNotReadyError, SourceRequestError
from apps.etl.models import RawTwse
from apps.etl.ratelimit import RateLimit, RateLimiter, RetryableError
from apps.market.calendar import NotATradingDayError, TradingDays
from apps.market.models import DailyPrice, DividendEvent

URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TRADING_DAY = date(2026, 10, 28)
FIXTURE = "twse/daily_price/mi_index_20261028.json"


@pytest.fixture
def payload(load_fixture: Callable[[str], Any]) -> dict[str, Any]:
    data: dict[str, Any] = load_fixture(FIXTURE)
    return data


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TwseAdapter:
    return TwseAdapter(october_2026, twse_limiter)


def _quotes_table(payload: dict[str, Any]) -> dict[str, Any]:
    table: dict[str, Any] = next(
        t for t in payload["tables"] if "每日收盤行情" in (t.get("title") or "")
    )
    return table


@responses.activate
@pytest.mark.django_db
def test_fetch_stores_raw_and_returns_source_rows(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json=payload)
    raw = adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    assert len(raw) == 7
    assert set(raw["date"]) == {TRADING_DAY}
    stored = RawTwse.objects.get()
    assert stored.dataset == "daily_price"
    assert stored.request_params == {"date": "20261028", "type": "ALLBUT0999", "response": "json"}
    assert stored.payload == payload
    sent = parse_qs(urlparse(responses.calls[0].request.url or "").query)
    assert {key: values[0] for key, values in sent.items()} == stored.request_params


@responses.activate
@pytest.mark.django_db
def test_normalize_maps_columns_and_keeps_shares_and_twd(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json=payload)
    frame = adapter.normalize("daily_price", adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY))
    rows = frame.set_index("stock_id")
    assert list(frame.columns) == [
        "stock_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "date",
    ]
    tsmc = rows.loc["2330"]
    assert (tsmc["open"], tsmc["close"]) == (Decimal("2420.00"), Decimal("2425.00"))
    assert tsmc["volume"] == 21_866_835
    assert tsmc["turnover"] == 52_940_678_675
    assert tsmc["date"] == TRADING_DAY


@responses.activate
@pytest.mark.django_db
def test_no_trade_rows_keep_volume_and_null_prices(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json=payload)
    rows = adapter.normalize(
        "daily_price", adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    ).set_index("stock_id")
    assert rows.loc["00647L", "close"] is None
    assert rows.loc["00647L", "volume"] == 464
    assert rows.loc["00682U", "volume"] == 0


@responses.activate
@pytest.mark.django_db
def test_upsert_twice_keeps_row_count(adapter: TwseAdapter, payload: dict[str, Any]) -> None:
    responses.get(URL, json=payload)
    responses.get(URL, json=payload)
    for _ in range(2):
        raw = adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
        assert adapter.upsert("daily_price", adapter.normalize("daily_price", raw)) == 6
    assert DailyPrice.objects.count() == 6
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_unpublished_trading_day_raises_not_ready_and_keeps_no_raw(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(URL, json=load_fixture("twse/daily_price/mi_index_20261025_no_data.json"))
    with pytest.raises(SourceNotReadyError):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    assert RawTwse.objects.count() == 0


@responses.activate
def test_non_trading_day_is_rejected_before_any_request(adapter: TwseAdapter) -> None:
    with pytest.raises(NotATradingDayError):
        adapter.fetch("daily_price", date(2026, 10, 25), date(2026, 10, 25))
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.django_db
def test_previous_day_answer_means_not_published_yet(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json={**payload, "date": "20261027"})
    with pytest.raises(SourceNotReadyError, match="still serves 20261027"):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("answered", ["20261029", "2026-10-28", ""])
def test_unexpected_answer_date_is_a_schema_change_and_keeps_raw(
    adapter: TwseAdapter, payload: dict[str, Any], answered: str
) -> None:
    responses.get(URL, json={**payload, "date": answered})
    with pytest.raises(SchemaChangedError, match="instead of"):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_unexpected_stat_is_a_schema_change_and_keeps_raw(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json={**payload, "stat": "ok"})
    with pytest.raises(SchemaChangedError, match="unexpected stat"):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_unit_change_fails_loudly_but_keeps_raw(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    changed = copy.deepcopy(payload)
    _quotes_table(changed)["hints"] = "單位：元、張"
    responses.get(URL, json=changed)
    with pytest.raises(SchemaChangedError, match="units"):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_renamed_column_fails_normalize(adapter: TwseAdapter, payload: dict[str, Any]) -> None:
    changed = copy.deepcopy(payload)
    table = _quotes_table(changed)
    table["fields"] = ["代號" if f == "證券代號" else f for f in table["fields"]]
    responses.get(URL, json=changed)
    raw = adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    with pytest.raises(SchemaChangedError, match="lost columns"):
        adapter.normalize("daily_price", raw)


@responses.activate
@pytest.mark.django_db
def test_garbled_numbers_fail_normalize(adapter: TwseAdapter, payload: dict[str, Any]) -> None:
    changed = copy.deepcopy(payload)
    _quotes_table(changed)["data"][0][2] = "N/A"
    responses.get(URL, json=changed)
    raw = adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    with pytest.raises(SchemaChangedError, match="count"):
        adapter.normalize("daily_price", raw)


@responses.activate
@pytest.mark.django_db
def test_throttle_then_success_waits_for_cooldown(
    october_2026: TradingDays, fake_sleep: Any, payload: dict[str, Any]
) -> None:
    limit = RateLimit(
        min_interval=3.0, max_retries=2, backoff_base=1.0, backoff_max=2.0, cooldown=7.0
    )
    adapter = TwseAdapter(october_2026, RateLimiter(limit, sleep=fake_sleep))
    responses.get(URL, status=429)
    responses.get(URL, json=payload)
    assert len(adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)) == 7
    assert 7.0 in fake_sleep.calls


@responses.activate
def test_server_errors_are_retried_then_raised(adapter: TwseAdapter) -> None:
    for _ in range(3):
        responses.get(URL, status=503)
    with pytest.raises(RetryableError):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    assert len(responses.calls) == 3


@responses.activate
def test_connection_errors_are_retryable(adapter: TwseAdapter) -> None:
    for _ in range(3):
        responses.get(URL, body=requests.ConnectionError("reset"))
    with pytest.raises(RetryableError):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)


@responses.activate
def test_other_request_failures_are_retryable(adapter: TwseAdapter) -> None:
    for _ in range(3):
        responses.get(URL, body=requests.exceptions.ChunkedEncodingError("cut"))
    with pytest.raises(RetryableError):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    assert len(responses.calls) == 3


@responses.activate
@pytest.mark.parametrize("status", [403, 404])
def test_client_errors_are_not_retried(adapter: TwseAdapter, status: int) -> None:
    responses.get(URL, status=status)
    with pytest.raises(SourceRequestError, match=str(status)):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    assert len(responses.calls) == 1


@responses.activate
def test_html_block_page_is_reported(adapter: TwseAdapter) -> None:
    responses.get(URL, body="<html>blocked</html>", content_type="text/html")
    with pytest.raises(SchemaChangedError, match="non-JSON"):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)


def test_interval_below_three_seconds_is_refused(october_2026: TradingDays) -> None:
    fast = RateLimit(
        min_interval=1.0, max_retries=0, backoff_base=1.0, backoff_max=1.0, cooldown=1.0
    )
    with pytest.raises(ImproperlyConfigured, match="3 seconds"):
        TwseAdapter(october_2026, RateLimiter(fast))


@responses.activate
@pytest.mark.django_db
def test_garbled_prices_fail_normalize(adapter: TwseAdapter, payload: dict[str, Any]) -> None:
    changed = copy.deepcopy(payload)
    _quotes_table(changed)["data"][0][8] = "abc"
    responses.get(URL, json=changed)
    raw = adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    with pytest.raises(SchemaChangedError, match="price"):
        adapter.normalize("daily_price", raw)


@responses.activate
@pytest.mark.django_db
def test_missing_quotes_table_is_a_schema_change(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    changed = copy.deepcopy(payload)
    changed["tables"] = [
        t for t in changed["tables"] if "每日收盤行情" not in (t.get("title") or "")
    ]
    responses.get(URL, json=changed)
    with pytest.raises(SchemaChangedError, match="no table"):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)


@responses.activate
def test_json_that_is_not_an_object_is_a_schema_change(adapter: TwseAdapter) -> None:
    responses.get(URL, json=[1, 2, 3])
    with pytest.raises(SchemaChangedError, match="not an object"):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("missing", ["data", "fields"])
def test_malformed_quotes_table_is_a_schema_change(
    adapter: TwseAdapter, payload: dict[str, Any], missing: str
) -> None:
    changed = copy.deepcopy(payload)
    del _quotes_table(changed)[missing]
    responses.get(URL, json=changed)
    with pytest.raises(SchemaChangedError, match="malformed"):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_foreign_currency_counters_are_dropped(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json=payload)
    raw = adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    assert "00636K" in set(raw["證券代號"])
    frame = adapter.normalize("daily_price", raw)
    assert "00636K" not in set(frame["stock_id"])
    assert len(frame) == 6


@pytest.mark.parametrize(
    ("code", "foreign"),
    [
        ("00636K", True),
        ("00625K", True),
        ("00999M", True),
        ("00999S", True),
        ("00999C", True),
        ("0050", False),
        ("00647L", False),
        ("00682U", False),
        ("2887Z1", False),
        ("2330", False),
    ],
)
def test_foreign_currency_etf_codes(code: str, foreign: bool) -> None:
    assert is_foreign_currency_etf(code) is foreign


@responses.activate
@pytest.mark.django_db
def test_refetch_keeps_adj_factor(adapter: TwseAdapter, payload: dict[str, Any]) -> None:
    responses.get(URL, json=payload)
    raw = adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    adapter.upsert("daily_price", adapter.normalize("daily_price", raw))
    DailyPrice.objects.filter(stock_id="2330").update(adj_factor=0.5)
    raw = adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    adapter.upsert("daily_price", adapter.normalize("daily_price", raw))
    assert DailyPrice.objects.get(stock_id="2330").adj_factor == 0.5


@responses.activate
@pytest.mark.django_db
def test_stored_payload_rebuilds_the_same_rows(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json=payload)
    fetched = adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
    rebuilt = quotes_frame(RawTwse.objects.get().payload, TRADING_DAY)
    pd.testing.assert_frame_equal(rebuilt, fetched)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"date": "20261027"}, SourceNotReadyError),
        ({"date": "2026102"}, SchemaChangedError),
        ({"stat": "查詢日期大於今日，請重新查詢!"}, SchemaChangedError),
    ],
)
def test_rerun_from_raw_rechecks_the_answer(
    payload: dict[str, Any], change: dict[str, str], error: type[Exception]
) -> None:
    with pytest.raises(error):
        quotes_frame({**payload, **change}, TRADING_DAY)


def test_empty_quotes_table_is_a_schema_change(payload: dict[str, Any]) -> None:
    changed = copy.deepcopy(payload)
    _quotes_table(changed)["data"] = []
    with pytest.raises(SchemaChangedError, match="no rows"):
        quotes_frame(changed, TRADING_DAY)


@responses.activate
@pytest.mark.django_db
def test_upsert_refreshes_factors_of_stocks_with_later_events(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    DividendEvent.objects.create(stock_id="2330", ex_date=date(2026, 10, 29), adj_factor=0.9)
    responses.get(URL, json=payload)
    raw = adapter.fetch("daily_price", date(2026, 10, 28), date(2026, 10, 28))
    adapter.upsert("daily_price", adapter.normalize("daily_price", raw))
    factors = dict(DailyPrice.objects.values_list("stock_id", "adj_factor"))
    assert factors["2330"] == pytest.approx(0.9)
    assert {code: f for code, f in factors.items() if code != "2330"} == dict.fromkeys(
        set(factors) - {"2330"}, 1.0
    )


@responses.activate
def test_non_standard_json_numbers_are_a_schema_change(adapter: TwseAdapter) -> None:
    responses.get(URL, body='{"stat": "OK", "value": NaN}', content_type="application/json")
    with pytest.raises(SchemaChangedError, match="NaN"):
        adapter.fetch("daily_price", TRADING_DAY, TRADING_DAY)
