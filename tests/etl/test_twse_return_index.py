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

URL = "https://www.twse.com.tw/rwd/zh/TAIEX/MFI94U"
DAY = date(2026, 10, 28)
FIXTURE = "twse/reference/mfi94u_202610.json"


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
def test_return_index_close_of_the_day(adapter: TwseAdapter, payload: dict[str, Any]) -> None:
    responses.get(URL, json=payload)
    frame = adapter.normalize("return_index", adapter.fetch("return_index", DAY, DAY))
    assert frame.to_dict("records") == [
        {"index_id": "TAIEX_TR", "date": DAY, "close": Decimal("108763.96"), "volume": None}
    ]
    stored = RawTwse.objects.get()
    assert stored.dataset == "return_index"
    assert stored.request_params == {"date": "20261001", "response": "json"}


@responses.activate
@pytest.mark.django_db
def test_return_index_of_2010_with_padded_dates(
    january_2010: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(URL, json=load_fixture("twse/reference/mfi94u_201001.json"))
    adapter = TwseAdapter(january_2010, twse_limiter)
    first = date(2010, 1, 4)
    frame = adapter.normalize("return_index", adapter.fetch("return_index", first, first))
    assert frame["close"].tolist() == [Decimal("10528.26")]


@responses.activate
@pytest.mark.django_db
def test_day_after_the_last_published_row_is_not_ready(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json=_without(payload, "115/10/30"))
    with pytest.raises(SourceNotReadyError, match="2026-10-30"):
        adapter.fetch("return_index", date(2026, 10, 30), date(2026, 10, 30))


@responses.activate
@pytest.mark.django_db
def test_month_without_rows_is_not_ready_and_not_kept(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json={**payload, "data": [], "total": 0})
    with pytest.raises(SourceNotReadyError, match=r"^TWSE MFI94U "):
        adapter.fetch("return_index", DAY, DAY)
    assert not RawTwse.objects.exists()


@responses.activate
@pytest.mark.django_db
def test_trading_day_missing_inside_the_month_is_a_schema_change(
    adapter: TwseAdapter, payload: dict[str, Any]
) -> None:
    responses.get(URL, json=_without(payload, "115/10/28"))
    with pytest.raises(SchemaChangedError, match="2026-10-28"):
        adapter.fetch("return_index", DAY, DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"fields": ["日期", "發行量加權股價指數"]}, "fields"),
        ({"stat": "異常"}, "stat"),
        ({"data": [["115/10/28", "--"]]}, "no close"),
        ({"data": [["115/10/28"]]}, "malformed"),
    ],
    ids=["fields", "stat", "no-close", "short-row"],
)
def test_layout_change_is_a_schema_change(
    adapter: TwseAdapter, payload: dict[str, Any], change: dict[str, Any], message: str
) -> None:
    responses.get(URL, json={**payload, **change})
    with pytest.raises(SchemaChangedError, match=message):
        adapter.normalize("return_index", adapter.fetch("return_index", DAY, DAY))


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_fetch_command_backfills_a_month_in_one_request_and_reruns_in_place(
    payload: dict[str, Any], october_2026: TradingDays, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    responses.get(URL, json=payload)
    IndexDaily.objects.create(index_id="TAIEX", date=DAY, close=Decimal("47035.58"))
    for _ in range(2):
        call_command(
            "fetch", source="twse", dataset="return_index", start="2026-10-01", end="2026-10-30"
        )
    assert len(responses.calls) == 2
    october = october_2026.between(date(2026, 10, 1), date(2026, 10, 30))
    stored = IndexDaily.objects.filter(index_id="TAIEX_TR").order_by("date")
    assert list(stored.values_list("date", flat=True)) == list(october)
    assert stored.get(date=DAY).close == Decimal("108763.9600")
    assert IndexDaily.objects.get(index_id="TAIEX").close == Decimal("47035.5800")
    assert list(EtlRun.objects.values_list("rows", flat=True)) == [len(october)] * 2
