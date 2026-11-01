import copy
import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import responses
import time_machine
from django.core.management import call_command

from apps.etl.adapters.yahoo import YahooAdapter
from apps.etl.errors import SchemaChangedError, SourceRequestError
from apps.etl.models import EtlRun, RawYahoo
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import NotATradingDayError, TradingDays
from apps.market.models import IndexDaily

BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
SERIES = {
    "sox": (f"{BASE}/%5ESOX", "yahoo/sox_202610.json", "SOX"),
    "tsm_adr": (f"{BASE}/TSM", "yahoo/tsm_202610.json", "TSM_ADR"),
    "usdtwd": (f"{BASE}/TWD%3DX", "yahoo/twd_x_202610.json", "USDTWD"),
}
DAY = date(2026, 10, 28)


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> YahooAdapter:
    return YahooAdapter(october_2026, twse_limiter)


def _payload(load_fixture: Callable[[str], Any], dataset: str) -> dict[str, Any]:
    return copy.deepcopy(load_fixture(SERIES[dataset][1]))


def _result(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = payload["chart"]["result"][0]
    return result


def _closes(
    adapter: YahooAdapter,
    payload: dict[str, Any],
    dataset: str = "sox",
    start: date = DAY,
    end: date = DAY,
) -> pd.DataFrame:
    responses.get(SERIES[dataset][0], json=payload)
    return adapter.normalize(dataset, adapter.fetch(dataset, start, end))


def _by_date(frame: pd.DataFrame) -> dict[date, Any]:
    return dict(zip(frame["date"], frame["close"], strict=True))


def _sent() -> dict[str, str]:
    query = urlparse(responses.calls[0].request.url or "").query
    return {key: values[0] for key, values in parse_qs(query).items()}


@responses.activate
@pytest.mark.django_db
def test_fetch_asks_for_daily_bars_around_the_day_and_keeps_raw(
    adapter: YahooAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = _payload(load_fixture, "sox")
    _closes(adapter, payload)
    first = int(datetime(2026, 10, 27, tzinfo=UTC).timestamp())
    last = int(datetime(2026, 10, 30, tzinfo=UTC).timestamp())
    assert _sent() == {"interval": "1d", "period1": str(first), "period2": str(last)}
    raw = RawYahoo.objects.get()
    assert (raw.dataset, raw.payload) == ("sox", payload)
    assert raw.request_params["symbol"] == "^SOX"


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("dataset", "close", "volume"),
    [
        ("sox", Decimal("11684.00"), None),
        ("tsm_adr", Decimal("439.51"), 8037400),
        ("usdtwd", Decimal("32.4255"), None),
    ],
)
def test_each_series_keeps_its_close_at_the_published_precision(
    adapter: YahooAdapter,
    load_fixture: Callable[[str], Any],
    dataset: str,
    close: Decimal,
    volume: int | None,
) -> None:
    frame = _closes(adapter, _payload(load_fixture, dataset), dataset)
    assert frame.to_dict("records") == [
        {"index_id": SERIES[dataset][2], "date": DAY, "close": close, "volume": volume}
    ]


@responses.activate
@pytest.mark.django_db
def test_days_taiwan_was_closed_are_written_with_the_next_session(
    adapter: YahooAdapter, load_fixture: Callable[[str], Any]
) -> None:
    day = date(2026, 10, 27)
    frame = _closes(adapter, _payload(load_fixture, "sox"), "sox", day, day)
    assert list(frame["date"]) == [date(2026, 10, 23), date(2026, 10, 26), day]


@responses.activate
@pytest.mark.django_db
def test_dates_stay_on_the_exchange_calendar(
    adapter: YahooAdapter, load_fixture: Callable[[str], Any]
) -> None:
    day = date(2026, 10, 12)
    frame = _closes(adapter, _payload(load_fixture, "usdtwd"), "usdtwd", day, day)
    assert _by_date(frame) == {
        date(2026, 10, 9): Decimal("31.9254"),
        date(2026, 10, 12): Decimal("31.9161"),
    }


@responses.activate
@pytest.mark.django_db
def test_a_range_from_the_calendar_start_writes_every_bar_in_it(
    adapter: YahooAdapter, load_fixture: Callable[[str], Any]
) -> None:
    frame = _closes(
        adapter, _payload(load_fixture, "tsm_adr"), "tsm_adr", date(2026, 10, 1), date(2026, 10, 30)
    )
    assert len(frame) == 22
    assert min(frame["date"]) == date(2026, 10, 1)
    assert max(frame["date"]) == date(2026, 10, 30)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("moment", "kept"),
    [(datetime(2026, 10, 30, 15, tzinfo=UTC), False), (datetime(2026, 10, 31, tzinfo=UTC), True)],
    ids=["session-open", "after-close"],
)
def test_a_bar_whose_session_is_still_open_is_left_for_a_later_run(
    adapter: YahooAdapter, load_fixture: Callable[[str], Any], moment: datetime, kept: bool
) -> None:
    payload = _payload(load_fixture, "sox")
    with time_machine.travel(moment, tick=False):
        frame = _closes(adapter, payload, start=date(2026, 10, 29), end=date(2026, 10, 30))
    assert (date(2026, 10, 30) in set(frame["date"])) is kept
    assert date(2026, 10, 29) in set(frame["date"])


@responses.activate
@pytest.mark.django_db
def test_a_bar_without_a_close_is_skipped_with_a_warning(
    adapter: YahooAdapter, load_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
) -> None:
    payload = _payload(load_fixture, "usdtwd")
    quote = _result(payload)["indicators"]["quote"][0]
    index = len(quote["close"]) - 3
    for field in quote:
        quote[field][index] = None
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.yahoo"):
        frame = _closes(adapter, payload, "usdtwd", start=date(2026, 10, 27))
    assert list(frame["date"]) == [date(2026, 10, 23), date(2026, 10, 26), date(2026, 10, 27)]
    assert "no close" in caplog.text


@responses.activate
@pytest.mark.django_db
def test_a_day_without_a_bar_adds_no_row(
    adapter: YahooAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = _payload(load_fixture, "sox")
    result = _result(payload)
    index = len(result["timestamp"]) - 3
    result["timestamp"].pop(index)
    for values in result["indicators"]["quote"][0].values():
        values.pop(index)
    assert _closes(adapter, payload).empty


@responses.activate
@pytest.mark.django_db
def test_a_range_without_any_bar_is_an_empty_answer(
    adapter: YahooAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = _payload(load_fixture, "sox")
    result = _result(payload)
    del result["timestamp"]
    result["indicators"] = {"quote": [{}], "adjclose": [{}]}
    frame = _closes(adapter, payload)
    assert frame.empty
    assert adapter.upsert("sox", frame) == 0


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p["chart"].__setitem__("error", {"code": "x"}), "error"),
        (lambda p: p["chart"].__setitem__("result", []), "result"),
        (lambda p: _result(p)["meta"].__setitem__("symbol", "^SPX"), "symbol"),
        (lambda p: _result(p)["meta"].__setitem__("exchangeTimezoneName", "Mars/Base"), "zone"),
        (lambda p: _result(p)["meta"].pop("currentTradingPeriod"), "trading period"),
        (lambda p: _result(p)["meta"].__setitem__("priceHint", "2"), "price hint"),
        (lambda p: _result(p)["timestamp"].pop(), "bars"),
        (lambda p: _result(p)["indicators"]["quote"][0]["close"].__setitem__(-3, -1.0), "close"),
        (lambda p: _result(p)["indicators"]["quote"][0]["close"].__setitem__(-3, "x"), "close"),
        (lambda p: _result(p)["timestamp"].__setitem__(-2, _result(p)["timestamp"][-3]), "dates"),
    ],
    ids=[
        "error",
        "no-result",
        "symbol",
        "zone",
        "no-period",
        "hint",
        "lengths",
        "negative",
        "word",
        "repeated",
    ],
)
def test_malformed_answers_are_schema_changes_and_keep_raw(
    adapter: YahooAdapter,
    load_fixture: Callable[[str], Any],
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    payload = _payload(load_fixture, "sox")
    change(payload)
    with pytest.raises(SchemaChangedError, match=message):
        _closes(adapter, payload)
    assert RawYahoo.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_an_unknown_symbol_is_a_request_error(adapter: YahooAdapter) -> None:
    responses.get(
        SERIES["sox"][0],
        status=404,
        json={"chart": {"result": None, "error": {"code": "Not Found"}}},
    )
    with pytest.raises(SourceRequestError, match="404"):
        adapter.fetch("sox", DAY, DAY)


@responses.activate
@pytest.mark.django_db
def test_upsert_is_idempotent_and_leaves_other_series_alone(
    adapter: YahooAdapter, load_fixture: Callable[[str], Any]
) -> None:
    IndexDaily.objects.create(index_id="TAIEX", date=DAY, close=Decimal("28000"), volume=1)
    for _ in range(2):
        adapter.upsert(
            "sox", _closes(adapter, _payload(load_fixture, "sox"), start=date(2026, 10, 27))
        )
    assert IndexDaily.objects.filter(index_id="SOX").count() == 4
    assert IndexDaily.objects.get(index_id="TAIEX").close == Decimal("28000")


@responses.activate
def test_non_trading_day_is_rejected_before_any_request(adapter: YahooAdapter) -> None:
    with pytest.raises(NotATradingDayError):
        list(adapter.units("sox", date(2026, 10, 23), date(2026, 10, 23)))
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
@pytest.mark.parametrize("dataset", sorted(SERIES))
def test_fetch_command_writes_each_series(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch, dataset: str
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    responses.get(SERIES[dataset][0], json=load_fixture(SERIES[dataset][1]))
    call_command("fetch", source="yahoo", dataset=dataset, date="2026-10-28")
    run = EtlRun.objects.get()
    assert (run.status, run.rows) == ("success", 1)
    assert IndexDaily.objects.filter(index_id=SERIES[dataset][2], date=DAY).exists()


@responses.activate
@pytest.mark.django_db
def test_an_adr_bar_without_volume_keeps_the_close(
    adapter: YahooAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = _payload(load_fixture, "tsm_adr")
    _result(payload)["indicators"]["quote"][0]["volume"][-3] = None
    frame = _closes(adapter, payload, "tsm_adr")
    assert frame.to_dict("records")[0]["volume"] is None


@responses.activate
@pytest.mark.django_db
def test_a_negative_adr_volume_is_a_schema_change(
    adapter: YahooAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = _payload(load_fixture, "tsm_adr")
    _result(payload)["indicators"]["quote"][0]["volume"][-3] = -5
    with pytest.raises(SchemaChangedError, match="volume"):
        _closes(adapter, payload, "tsm_adr")
