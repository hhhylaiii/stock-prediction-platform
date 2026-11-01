import logging
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import responses
from django.core.management import call_command

from apps.etl.adapters.twse import TwseAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import EtlRun, RawTwse
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
@pytest.mark.parametrize("text", ["inf", "-Infinity", "nan", "1e400"])
def test_valuation_non_finite_number_is_a_schema_change(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any], text: str
) -> None:
    payload = load_fixture(VALUATION)
    next(row for row in payload["data"] if row[0] == "2330")[5] = text
    responses.get(VALUATION_URL, json=payload)
    with pytest.raises(SchemaChangedError, match="ratio"):
        adapter.normalize("daily_valuation", adapter.fetch("daily_valuation", DAY, DAY))
    assert RawTwse.objects.count() == 1


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


@responses.activate
@pytest.mark.django_db
def test_first_snapshot_writes_every_local_currency_count(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    assert _load_shares(adapter, load_fixture(SHARES[DAY]), DAY) == 3
    assert _shares_in_db() == {
        ("0050", DAY): 23_267_000_000,
        ("2330", DAY): 25_932_370_067,
        ("2368", DAY): 517_016_714,
    }
    assert set(SharesOutstanding.objects.values_list("reason", flat=True)) == {"unknown"}


@responses.activate
@pytest.mark.django_db
def test_later_snapshot_writes_only_changed_and_new_counts(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _load_shares(adapter, load_fixture(SHARES[DAY]), DAY)
    assert _load_shares(adapter, load_fixture(SHARES[NEXT]), NEXT) == 3
    later = {key: shares for key, shares in _shares_in_db().items() if key[1] == NEXT}
    assert later == {
        ("0050", NEXT): 23_455_000_000,
        ("2368", NEXT): 532_451_280,
        ("7867", NEXT): 106_200_000,
    }


@responses.activate
@pytest.mark.django_db
def test_rerunning_a_day_changes_nothing(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _load_shares(adapter, load_fixture(SHARES[DAY]), DAY)
    for _ in range(2):
        _load_shares(adapter, load_fixture(SHARES[NEXT]), NEXT)
    assert SharesOutstanding.objects.count() == 6


@responses.activate
@pytest.mark.django_db
def test_count_matching_an_earlier_backfill_row_adds_nothing(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    SharesOutstanding.objects.create(
        stock_id="2330", effective_date=date(2026, 5, 26), shares=25_932_370_067
    )
    _load_shares(adapter, load_fixture(SHARES[DAY]), DAY)
    assert ("2330", DAY) not in _shares_in_db()


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("older", "newer", "written"),
    [(500_000_000, 517_016_714, False), (517_016_714, 500_000_000, True)],
    ids=["latest-matches", "only-older-matches"],
)
def test_comparison_uses_the_latest_earlier_row(
    adapter: TwseAdapter,
    load_fixture: Callable[[str], Any],
    older: int,
    newer: int,
    written: bool,
) -> None:
    SharesOutstanding.objects.create(stock_id="2368", effective_date=date(2026, 5, 1), shares=older)
    SharesOutstanding.objects.create(
        stock_id="2368", effective_date=date(2026, 10, 1), shares=newer
    )
    _load_shares(adapter, load_fixture(SHARES[DAY]), DAY)
    assert (("2368", DAY) in _shares_in_db()) is written


@responses.activate
@pytest.mark.django_db
def test_negative_count_is_a_schema_change_and_keeps_raw(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(SHARES[DAY])
    next(row for row in payload["data"] if row[0] == "2368")[3] = "-5"
    with pytest.raises(SchemaChangedError, match="-5"):
        _load_shares(adapter, payload, DAY)
    assert RawTwse.objects.count() == 1
    assert not SharesOutstanding.objects.exists()


@responses.activate
@pytest.mark.django_db
def test_range_compares_each_day_with_the_one_before(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    frames = []
    for day in (DAY, NEXT):
        responses.get(HOLDING_URL, json=load_fixture(SHARES[day]))
        frames.append(adapter.fetch("shares_outstanding", day, day))
    rows = adapter.normalize("shares_outstanding", pd.concat(frames, ignore_index=True))
    written = {(row["stock_id"], row["effective_date"]) for row in rows.to_dict("records")}
    assert ("2330", NEXT) not in written
    assert {("2330", DAY), ("2368", DAY), ("2368", NEXT), ("7867", NEXT)} <= written


@responses.activate
@pytest.mark.django_db
def test_zero_count_is_skipped_with_a_warning(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
) -> None:
    payload = load_fixture(SHARES[DAY])
    next(row for row in payload["data"] if row[0] == "2368")[3] = "0"
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.twse_tables"):
        _load_shares(adapter, payload, DAY)
    assert ("2368", DAY) not in _shares_in_db()
    assert "2368" in caplog.text


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_fetch_command_writes_share_changes(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    for day in (DAY, NEXT):
        responses.get(HOLDING_URL, json=load_fixture(SHARES[day]))
        call_command("fetch", source="twse", dataset="shares_outstanding", date=day.isoformat())
    runs = EtlRun.objects.filter(dataset="shares_outstanding").order_by("run_id")
    assert list(runs.values_list("status", "rows")) == [("success", 3), ("success", 3)]
    assert SharesOutstanding.objects.filter(stock_id="2368").count() == 2
