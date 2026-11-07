import logging
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.parse import parse_qs

import pytest
import responses
from django.core.management import call_command

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import EtlRun, RawTpex
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import DailyValuation, ShareChangeReason, SharesOutstanding

BASE = "https://www.tpex.org.tw/www/zh-tw"
VALUATION_URL = f"{BASE}/afterTrading/peQryDate"
HOLDING_URL = f"{BASE}/insti/qfii"
VALUATION = "tpex/daily_valuation/pe_20261028.json"
HOLDING = "tpex/foreign_holding/qfii_20261028.json"
DAY = date(2026, 10, 28)
NEXT = date(2026, 10, 29)


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TpexAdapter:
    return TpexAdapter(october_2026, twse_limiter)


def _sent(call_index: int) -> dict[str, str]:
    body = responses.calls[call_index].request.body or ""
    text = body.decode() if isinstance(body, bytes) else body
    return {key: values[0] for key, values in parse_qs(text).items()}


def _rows(frame: Any) -> dict[str, dict[str, Any]]:
    return {
        str(row["stock_id"]): {str(key): value for key, value in row.items()}
        for row in frame.to_dict("records")
    }


def _valuation(adapter: TpexAdapter, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    responses.post(VALUATION_URL, json=payload)
    return _rows(adapter.normalize("daily_valuation", adapter.fetch("daily_valuation", DAY, DAY)))


def _load_shares(adapter: TpexAdapter, payload: dict[str, Any], day: date) -> int:
    responses.post(HOLDING_URL, json={**payload, "date": f"{day:%Y%m%d}"})
    raw = adapter.fetch("shares_outstanding", day, day)
    return adapter.upsert("shares_outstanding", adapter.normalize("shares_outstanding", raw))


def _shares_in_db() -> dict[tuple[str, date], int]:
    return {
        (stock_id, effective): shares
        for stock_id, effective, shares in SharesOutstanding.objects.values_list(
            "stock_id", "effective_date", "shares"
        )
    }


def _set_count(payload: dict[str, Any], code: str, count: str) -> dict[str, Any]:
    next(row for row in payload["tables"][0]["data"] if row[1] == code)[3] = count
    return payload


@responses.activate
@pytest.mark.django_db
def test_valuation_posts_the_day_and_stores_raw(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(VALUATION)
    _valuation(adapter, payload)
    assert _sent(0) == {"date": "2026/10/28", "response": "json"}
    assert RawTpex.objects.get().payload == payload


@responses.activate
@pytest.mark.django_db
def test_valuation_keeps_ratios_and_turns_yield_percent_into_a_fraction(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    rows = _valuation(adapter, load_fixture(VALUATION))
    assert rows["6488"] == {
        "stock_id": "6488",
        "date": DAY,
        "pe": 41.21,
        "pb": 4.20,
        "dividend_yield": pytest.approx(0.0091),
    }
    assert (rows["1569"]["pe"], rows["1569"]["pb"]) == (None, 4.81)
    assert rows["2064"]["dividend_yield"] == 0.0
    assert set(rows) == {"1240", "1569", "2064", "3293", "6488"}


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("column", "text"),
    [(2, "-3.10"), (6, "0.00"), (5, "100.00"), (5, "-0.50")],
    ids=["pe", "pb", "yield-whole", "yield-negative"],
)
def test_valuation_out_of_range_is_dropped_with_a_warning(
    adapter: TpexAdapter,
    load_fixture: Callable[[str], Any],
    caplog: pytest.LogCaptureFixture,
    column: int,
    text: str,
) -> None:
    payload = load_fixture(VALUATION)
    next(row for row in payload["tables"][0]["data"] if row[0] == "6488")[column] = text
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.exchange_tables"):
        rows = _valuation(adapter, payload)
    assert rows["6488"][{2: "pe", 5: "dividend_yield", 6: "pb"}[column]] is None
    assert "TPEx afterTrading/peQryDate" in caplog.text


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("column", [2, 5, 6], ids=["pe", "yield", "pb"])
@pytest.mark.parametrize("text", ["虧損", "inf", "-", "", "NULL", "Null"])
def test_valuation_word_in_a_ratio_column_is_a_schema_change(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], text: str, column: int
) -> None:
    payload = load_fixture(VALUATION)
    payload["tables"][0]["data"][0][column] = text
    with pytest.raises(SchemaChangedError, match="ratio"):
        _valuation(adapter, payload)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("column", [2, 5, 6], ids=["pe", "yield", "pb"])
def test_valuation_not_available_in_any_ratio_column_is_missing(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], column: int
) -> None:
    payload = load_fixture(VALUATION)
    next(row for row in payload["tables"][0]["data"] if row[0] == "6488")[column] = "N/A"
    rows = _valuation(adapter, payload)
    assert rows["6488"][{2: "pe", 5: "dividend_yield", 6: "pb"}[column]] is None


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("column", [2, 5, 6], ids=["pe", "yield", "pb"])
@pytest.mark.parametrize("value", [None, "null"], ids=["json-null", "null-text"])
def test_valuation_null_in_any_ratio_column_is_missing(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], column: int, value: str | None
) -> None:
    payload = load_fixture(VALUATION)
    next(row for row in payload["tables"][0]["data"] if row[0] == "6488")[column] = value
    rows = _valuation(adapter, payload)
    assert rows["6488"][{2: "pe", 5: "dividend_yield", 6: "pb"}[column]] is None
    assert rows["1240"]["pe"] == pytest.approx(10.02)


@responses.activate
@pytest.mark.django_db
def test_valuation_without_the_percent_column_is_a_schema_change(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(VALUATION)
    payload["tables"][0]["fields"][5] = "殖利率"
    responses.post(VALUATION_URL, json=payload)
    with pytest.raises(SchemaChangedError, match="殖利率"):
        adapter.fetch("daily_valuation", DAY, DAY)
    assert RawTpex.objects.count() == 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("url", "dataset", "fixture"),
    [
        (VALUATION_URL, "daily_valuation", "tpex/daily_valuation/pe_20261101_empty.json"),
        (HOLDING_URL, "shares_outstanding", "tpex/foreign_holding/qfii_20261101_empty.json"),
    ],
)
def test_unpublished_day_is_not_ready(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], url: str, dataset: str, fixture: str
) -> None:
    responses.post(url, json=load_fixture(fixture))
    with pytest.raises(SourceNotReadyError):
        adapter.fetch(dataset, DAY, DAY)
    assert RawTpex.objects.count() == 0


@responses.activate
@pytest.mark.django_db
def test_valuation_upsert_is_idempotent(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    for _ in range(2):
        responses.post(VALUATION_URL, json=load_fixture(VALUATION))
        raw = adapter.fetch("daily_valuation", DAY, DAY)
        assert adapter.upsert("daily_valuation", adapter.normalize("daily_valuation", raw)) == 5
    assert DailyValuation.objects.count() == 5


@responses.activate
@pytest.mark.django_db
def test_first_snapshot_writes_every_count_including_unquoted_codes(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    assert _load_shares(adapter, load_fixture(HOLDING), DAY) == 5
    assert _shares_in_db() == {
        ("1240", DAY): 46_443_992,
        ("2064", DAY): 69_330_000,
        ("3293", DAY): 281_801_560,
        ("4804", DAY): 71_991_920,
        ("6488", DAY): 528_113_725,
    }
    assert set(SharesOutstanding.objects.values_list("reason", flat=True)) == {
        ShareChangeReason.UNKNOWN.value
    }


@responses.activate
@pytest.mark.django_db
def test_later_snapshot_writes_only_changed_counts(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _load_shares(adapter, load_fixture(HOLDING), DAY)
    assert (
        _load_shares(adapter, _set_count(load_fixture(HOLDING), "6488", "530,000,000"), NEXT) == 1
    )
    assert _shares_in_db()[("6488", NEXT)] == 530_000_000
    assert SharesOutstanding.objects.count() == 6


@responses.activate
@pytest.mark.django_db
def test_count_matching_an_earlier_backfill_row_adds_nothing(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    SharesOutstanding.objects.create(
        stock_id="6488", effective_date=date(2026, 6, 1), shares=528_113_725, reason="unknown"
    )
    _load_shares(adapter, load_fixture(HOLDING), DAY)
    assert ("6488", DAY) not in _shares_in_db()


@responses.activate
@pytest.mark.django_db
def test_later_day_run_first_leaves_no_repeated_row(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _load_shares(adapter, load_fixture(HOLDING), NEXT)
    _load_shares(adapter, load_fixture(HOLDING), DAY)
    assert sorted(day for code, day in _shares_in_db() if code == "6488") == [DAY]


@responses.activate
@pytest.mark.django_db
def test_zero_count_is_skipped_with_a_warning(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.exchange_tables"):
        _load_shares(adapter, _set_count(load_fixture(HOLDING), "6488", "0"), DAY)
    assert ("6488", DAY) not in _shares_in_db()
    assert "6488" in caplog.text


@responses.activate
@pytest.mark.django_db
def test_negative_count_is_a_schema_change_and_keeps_raw(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    with pytest.raises(SchemaChangedError, match="count"):
        _load_shares(adapter, _set_count(load_fixture(HOLDING), "6488", "-1"), DAY)
    assert RawTpex.objects.count() == 1


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_fetch_command_writes_share_changes(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    responses.post(HOLDING_URL, json=load_fixture(HOLDING))
    for _ in range(2):
        call_command("fetch", source="tpex", dataset="shares_outstanding", date="2026-10-28")
    assert SharesOutstanding.objects.count() == 5
    runs = EtlRun.objects.order_by("run_id").values_list("status", "rows")
    assert list(runs) == [("success", 5), ("success", 5)]
