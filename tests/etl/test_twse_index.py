from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
import responses
from django.core.management import call_command

from apps.etl.adapters.twse import TwseAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import EtlRun, RawTwse
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import IndexDaily

URL = "https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK"
DAY = date(2026, 10, 28)
FIXTURE = "twse/reference/fmtqik_202610.json"


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TwseAdapter:
    return TwseAdapter(october_2026, twse_limiter)


@pytest.fixture
def payload(load_fixture: Callable[[str], Any]) -> dict[str, Any]:
    data: dict[str, Any] = load_fixture(FIXTURE)
    return data


def _without(payload: dict[str, Any], roc_day: str) -> dict[str, Any]:
    return {**payload, "data": [row for row in payload["data"] if row[0] != roc_day]}


@responses.activate
@pytest.mark.django_db
def test_taiex_close_and_market_volume_of_the_day(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json=payload)
    frame = adapter.normalize("index_daily", adapter.fetch("index_daily", DAY, DAY))
    assert frame.to_dict("records") == [
        {"index_id": "TAIEX", "date": DAY, "close": Decimal("47035.58"), "volume": 9_000_687_119}
    ]
    stored = RawTwse.objects.get()
    assert stored.dataset == "index_daily"
    assert stored.request_params == {"date": "20261001", "response": "json"}


@responses.activate
@pytest.mark.django_db
def test_day_after_the_last_published_row_is_not_ready(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json=_without(payload, "115/10/30"))
    with pytest.raises(SourceNotReadyError, match="2026-10-30"):
        adapter.fetch("index_daily", date(2026, 10, 30), date(2026, 10, 30))
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_trading_day_missing_inside_the_month_is_a_schema_change(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json=_without(payload, "115/10/28"))
    with pytest.raises(SchemaChangedError, match="2026-10-28"):
        adapter.fetch("index_daily", DAY, DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("row", "message"),
    [
        (["115/09/30", "1", "1", "1", "47,000.00", "1.00"], "another month"),
        (["115/10/28", "1", "1", "1", "--", "0.00"], "no TAIEX close"),
        (["115/10/29", "1", "1"], "malformed"),
    ],
    ids=["other-month", "no-close", "short-row"],
)
def test_bad_monthly_rows_are_a_schema_change(
    adapter: TwseAdapter, payload: dict[str, Any], row: list[str], message: str
) -> None:
    responses.get(URL, json={**payload, "data": [*_without(payload, row[0])["data"], row]})
    with pytest.raises(SchemaChangedError, match=message):
        adapter.normalize("index_daily", adapter.fetch("index_daily", DAY, DAY))


@responses.activate
@pytest.mark.django_db
def test_month_without_data_is_not_ready(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(URL, json=load_fixture("twse/reference/fmtqik_202611.json"))
    with pytest.raises(SourceNotReadyError, match=r"^TWSE FMTQIK "):
        adapter.fetch("index_daily", DAY, DAY)
    assert RawTwse.objects.count() == 0


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    "change",
    [{"hints": "單位：元、張"}, {"fields": ["日期", "成交股數", "成交金額", "成交筆數", "指數"]}],
    ids=["units", "fields"],
)
def test_layout_change_is_a_schema_change(
    adapter: TwseAdapter, payload: dict[str, Any], change: dict[str, Any]
) -> None:
    responses.get(URL, json={**payload, **change})
    with pytest.raises(SchemaChangedError):
        adapter.fetch("index_daily", DAY, DAY)


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_fetch_command_writes_taiex_once_when_rerun(
    payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    responses.get(URL, json=payload)
    for _ in range(2):
        call_command("fetch", source="twse", dataset="index_daily", date="2026-10-28")
    taiex = IndexDaily.objects.get()
    assert (taiex.index_id, taiex.date, taiex.close) == ("TAIEX", DAY, Decimal("47035.5800"))
    assert list(EtlRun.objects.values_list("status", "rows")) == [("success", 1)] * 2
