from collections.abc import Callable
from datetime import date
from decimal import Decimal
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
from apps.market.models import IndexDaily

BASE = "https://www.tpex.org.tw/www/zh-tw"
DAY = date(2026, 10, 28)
INDEX = f"{BASE}/indexInfo/inx"


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TpexAdapter:
    return TpexAdapter(october_2026, twse_limiter)


def _sent(call_index: int) -> dict[str, str]:
    body = responses.calls[call_index].request.body or ""
    text = body.decode() if isinstance(body, bytes) else body
    return {key: values[0] for key, values in parse_qs(text).items()}


@responses.activate
@pytest.mark.django_db
def test_tpex_index_close_without_volume(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(INDEX, json=load_fixture("tpex/index_daily/inx_202610.json"))
    frame = adapter.normalize("index_daily", adapter.fetch("index_daily", DAY, DAY))
    assert frame.to_dict("records") == [
        {"index_id": "TPEX", "date": DAY, "close": Decimal("394.61"), "volume": None}
    ]
    assert _sent(0) == {"date": "2026/10/01", "response": "json"}


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("drop", "day", "error"),
    [
        ("2026/10/30", date(2026, 10, 30), SourceNotReadyError),
        ("2026/10/28", DAY, SchemaChangedError),
    ],
    ids=["after-last-row", "gap-inside-month"],
)
def test_tpex_index_missing_day(
    adapter: TpexAdapter,
    load_fixture: Callable[[str], Any],
    drop: str,
    day: date,
    error: type[Exception],
) -> None:
    payload = load_fixture("tpex/index_daily/inx_202610.json")
    table = payload["tables"][0]
    table["data"] = [row for row in table["data"] if row[0] != drop]
    table["totalCount"] = len(table["data"])
    responses.post(INDEX, json=payload)
    with pytest.raises(error, match=day.isoformat()):
        adapter.fetch("index_daily", day, day)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t["fields"].__setitem__(4, "收盤"), "fields"),
        (lambda t: t["data"][0].__setitem__(0, "115/10/01"), "date"),
        (lambda t: t["data"][-1].__setitem__(4, "---"), "close"),
    ],
    ids=["renamed", "roc-date", "no-close"],
)
def test_index_layout_change_is_a_schema_change(
    adapter: TpexAdapter,
    load_fixture: Callable[[str], Any],
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    payload = load_fixture("tpex/index_daily/inx_202610.json")
    change(payload["tables"][0])
    responses.post(INDEX, json=payload)
    day = date(2026, 10, 30)
    with pytest.raises(SchemaChangedError, match=message):
        adapter.normalize("index_daily", adapter.fetch("index_daily", day, day))


@responses.activate
@pytest.mark.django_db
def test_unpublished_month_is_not_ready_and_keeps_no_raw(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(INDEX, json=load_fixture("tpex/index_daily/inx_202611_empty.json"))
    with pytest.raises(SourceNotReadyError):
        adapter.fetch("index_daily", DAY, DAY)
    assert RawTpex.objects.count() == 0


@responses.activate
@pytest.mark.django_db
def test_previous_month_answer_means_not_published_yet(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("tpex/index_daily/inx_202610.json")
    responses.post(INDEX, json={**payload, "date": "20260901"})
    with pytest.raises(SourceNotReadyError, match="still serves 20260901"):
        adapter.fetch("index_daily", DAY, DAY)
    assert RawTpex.objects.count() == 1


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_fetch_command_backfills_a_month_in_one_request(
    load_fixture: Callable[[str], Any], october_2026: TradingDays, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    responses.post(INDEX, json=load_fixture("tpex/index_daily/inx_202610.json"))
    call_command(
        "fetch", source="tpex", dataset="index_daily", start="2026-10-01", end="2026-10-30"
    )
    assert len(responses.calls) == 1
    october = october_2026.between(date(2026, 10, 1), date(2026, 10, 30))
    stored = IndexDaily.objects.filter(index_id="TPEX").order_by("date")
    assert list(stored.values_list("date", flat=True)) == list(october)
    run = EtlRun.objects.get()
    assert (run.rows, run.checkpoint) == (len(october), {"last_date": "2026-10-30"})
