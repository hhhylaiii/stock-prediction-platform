from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs

import pytest
import responses

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import RawTpex
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, DividendEvent

BASE = "https://www.tpex.org.tw/www/zh-tw"
DAY = date(2026, 10, 28)
OTC = f"{BASE}/afterTrading/otc"
QUOTES = f"{BASE}/afterTrading/dailyQuotes"
QUOTE_FIXTURES = (
    "tpex/daily_price/otc_20261028.json",
    "tpex/daily_price/daily_quotes_20261028.json",
)


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TpexAdapter:
    return TpexAdapter(october_2026, twse_limiter)


@pytest.fixture
def serve_quotes(load_fixture: Callable[[str], Any]) -> Callable[[], None]:
    def _serve() -> None:
        responses.post(OTC, json=load_fixture(QUOTE_FIXTURES[0]))
        responses.post(QUOTES, json=load_fixture(QUOTE_FIXTURES[1]))

    return _serve


def _sent(call_index: int) -> dict[str, str]:
    body = responses.calls[call_index].request.body or ""
    text = body.decode() if isinstance(body, bytes) else body
    return {key: values[0] for key, values in parse_qs(text).items()}


def _load(adapter: TpexAdapter) -> dict[str, dict[str, Any]]:
    frame = adapter.normalize("daily_price", adapter.fetch("daily_price", DAY, DAY))
    assert set(frame["date"]) == {DAY}
    return {
        str(row["stock_id"]): {str(key): value for key, value in row.items()}
        for row in frame.to_dict("records")
    }


@responses.activate
@pytest.mark.django_db
def test_daily_price_posts_both_tables_and_keeps_both_raw(
    adapter: TpexAdapter, serve_quotes: Callable[[], None]
) -> None:
    serve_quotes()
    adapter.fetch("daily_price", DAY, DAY)
    assert [_sent(0), _sent(1)] == [
        {"date": "2026/10/28", "type": "EW", "response": "json"},
        {"date": "2026/10/28", "response": "json"},
    ]
    stored = RawTpex.objects.filter(dataset="daily_price").order_by("id")
    assert [row.request_params for row in stored] == [_sent(0), _sent(1)]


@responses.activate
@pytest.mark.django_db
def test_daily_price_takes_volume_with_odd_lots_and_skips_warrants(
    adapter: TpexAdapter, serve_quotes: Callable[[], None]
) -> None:
    serve_quotes()
    rows = _load(adapter)
    assert rows["6488"] == {
        "stock_id": "6488",
        "date": DAY,
        "open": Decimal("842.00"),
        "high": Decimal("860.00"),
        "low": Decimal("839.00"),
        "close": Decimal("849.00"),
        "volume": 6_892_320,
        "turnover": 5_852_153_367,
    }
    assert rows["1240"]["volume"] == 5_654
    assert set(rows) == {"006201", "1240", "2064", "3293", "6488"}


@responses.activate
@pytest.mark.django_db
def test_odd_lot_only_day_keeps_volume_and_null_prices(
    adapter: TpexAdapter, serve_quotes: Callable[[], None]
) -> None:
    serve_quotes()
    quiet = _load(adapter)["2064"]
    assert (quiet["open"], quiet["close"], quiet["volume"]) == (None, None, 203)


@responses.activate
@pytest.mark.django_db
def test_managed_stocks_in_the_ew_list_come_from_the_managed_table(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    otc, quotes = (load_fixture(name) for name in QUOTE_FIXTURES)
    listed, managed = quotes["tables"]
    managed["data"] = [row for row in listed["data"] if row[0] == "1240"]
    listed["data"] = [row for row in listed["data"] if row[0] != "1240"]
    managed["totalCount"], listed["totalCount"] = 1, len(listed["data"])
    responses.post(OTC, json=otc)
    responses.post(QUOTES, json=quotes)
    assert _load(adapter)["1240"]["volume"] == 5_654


@responses.activate
@pytest.mark.django_db
def test_ew_code_missing_from_daily_quotes_is_a_schema_change(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    quotes = load_fixture(QUOTE_FIXTURES[1])
    table = quotes["tables"][0]
    table["data"] = [row for row in table["data"] if row[0] != "6488"]
    table["totalCount"] = len(table["data"])
    responses.post(OTC, json=load_fixture(QUOTE_FIXTURES[0]))
    responses.post(QUOTES, json=quotes)
    with pytest.raises(SchemaChangedError, match="6488"):
        adapter.fetch("daily_price", DAY, DAY)


@responses.activate
@pytest.mark.django_db
def test_managed_stock_missing_from_the_ew_list_is_a_schema_change(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    otc, quotes = (load_fixture(name) for name in QUOTE_FIXTURES)
    listed, managed = quotes["tables"]
    managed["data"] = [["6131", *row[1:]] for row in listed["data"] if row[0] == "3293"]
    managed["totalCount"] = 1
    responses.post(OTC, json=otc)
    responses.post(QUOTES, json=quotes)
    with pytest.raises(SchemaChangedError, match="6131"):
        adapter.fetch("daily_price", DAY, DAY)


def _truncate_ew(otc: dict[str, Any], quotes: dict[str, Any]) -> None:
    otc["tables"][0]["data"].pop()


def _rename_ew_code(otc: dict[str, Any], quotes: dict[str, Any]) -> None:
    otc["tables"][0]["fields"][0] = "股票代號"


def _drop_volume(otc: dict[str, Any], quotes: dict[str, Any]) -> None:
    quotes["tables"][0]["fields"][8] = "成交張數"


def _repeat_in_managed(otc: dict[str, Any], quotes: dict[str, Any]) -> None:
    listed, managed = quotes["tables"]
    managed["data"] = [row for row in listed["data"] if row[0] == "6488"]
    managed["totalCount"] = 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_truncate_ew, "5 of 6 rows"),
        (_rename_ew_code, "fields"),
        (_drop_volume, "成交股數"),
        (_repeat_in_managed, "repeat 6488"),
    ],
    ids=["ew-truncated", "ew-renamed", "quotes-lost-volume", "quotes-repeat"],
)
def test_quotes_layout_change_is_a_schema_change_and_keeps_raw(
    adapter: TpexAdapter,
    load_fixture: Callable[[str], Any],
    change: Callable[[dict[str, Any], dict[str, Any]], None],
    message: str,
) -> None:
    otc, quotes = (load_fixture(name) for name in QUOTE_FIXTURES)
    change(otc, quotes)
    responses.post(OTC, json=otc)
    responses.post(QUOTES, json=quotes)
    with pytest.raises(SchemaChangedError, match=message):
        adapter.fetch("daily_price", DAY, DAY)
    assert RawTpex.objects.count() == 2


@responses.activate
@pytest.mark.django_db
def test_daily_price_upsert_is_idempotent_and_refreshes_adjustment(
    adapter: TpexAdapter, serve_quotes: Callable[[], None]
) -> None:
    DividendEvent.objects.create(stock_id="6488", ex_date=date(2026, 10, 29), adj_factor=0.5)
    for _ in range(2):
        serve_quotes()
        frame = adapter.normalize("daily_price", adapter.fetch("daily_price", DAY, DAY))
        assert adapter.upsert("daily_price", frame) == 5
    assert DailyPrice.objects.count() == 5
    assert DailyPrice.objects.get(stock_id="6488", date=DAY).adj_factor == 0.5


@responses.activate
@pytest.mark.django_db
def test_unpublished_quotes_are_not_ready(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(OTC, json=load_fixture("tpex/daily_price/otc_20261101_empty.json"))
    with pytest.raises(SourceNotReadyError):
        adapter.fetch("daily_price", DAY, DAY)
    assert RawTpex.objects.count() == 0


@responses.activate
@pytest.mark.django_db
def test_daily_quotes_still_empty_after_the_ew_list_is_not_ready(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(OTC, json=load_fixture(QUOTE_FIXTURES[0]))
    responses.post(QUOTES, json=load_fixture("tpex/daily_price/daily_quotes_20261101_empty.json"))
    with pytest.raises(SourceNotReadyError):
        adapter.fetch("daily_price", DAY, DAY)
    assert list(RawTpex.objects.values_list("request_params__type", flat=True)) == ["EW"]


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("stale", [0, 1], ids=["ew-list", "daily-quotes"])
def test_quotes_of_the_previous_day_mean_not_published_yet(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], stale: int
) -> None:
    payloads = [load_fixture(name) for name in QUOTE_FIXTURES]
    payloads[stale]["date"] = "20261027"
    responses.post(OTC, json=payloads[0])
    responses.post(QUOTES, json=payloads[1])
    with pytest.raises(SourceNotReadyError, match="still serves 20261027"):
        adapter.fetch("daily_price", DAY, DAY)
    assert RawTpex.objects.count() == 2
