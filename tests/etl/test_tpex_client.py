from collections.abc import Callable
from typing import Any

import pytest
import responses
from django.core.exceptions import ImproperlyConfigured
from responses import matchers

from apps.etl.adapters.base import SchemaChangedError, SourceRequestError
from apps.etl.adapters.tpex import TpexClient
from apps.etl.models import RawTpex
from apps.etl.ratelimit import RateLimit, RateLimiter

URL = "https://www.tpex.org.tw/www/zh-tw/company/deListed"
LIMIT = RateLimit(min_interval=3.0, max_retries=1, backoff_base=1.0, backoff_max=1.0, cooldown=1.0)


def _client(page_size: int = 1000) -> TpexClient:
    limiter = RateLimiter(LIMIT, sleep=lambda seconds: None)
    return TpexClient(limiter, "https://www.tpex.org.tw", 30.0, page_size=page_size)


def _page(payload: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    table = payload["tables"][0]
    return {**payload, "tables": [{**table, "data": table["data"][start:stop]}]}


def _matches(offset: int, size: int) -> list[Any]:
    return [matchers.urlencoded_params_matcher(_form(offset, size), allow_blank=True)]


def _form(offset: int, size: int) -> dict[str, str]:
    return {
        "code": "",
        "date": "ALL",
        "reason": "-1",
        "response": "json",
        "paging-offset": str(offset),
        "paging-size": str(size),
    }


@responses.activate
@pytest.mark.django_db
def test_delisted_pages_until_every_row_is_read(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture("tpex/delisted.json")
    responses.post(
        URL,
        json=_page(payload, 0, 5),
        match=_matches(0, 5),
    )
    responses.post(
        URL,
        json=_page(payload, 5, 8),
        match=_matches(5, 5),
    )
    table = _client(page_size=5).delisted()
    assert len(table["data"]) == 8
    assert table["fields"] == payload["tables"][0]["fields"]
    assert RawTpex.objects.filter(dataset="delisted").count() == 2


@responses.activate
@pytest.mark.django_db
def test_an_empty_page_stops_paging(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture("tpex/delisted.json")
    responses.post(URL, json=_page(payload, 0, 5))
    responses.post(URL, json=_page(payload, 0, 0))
    assert len(_client(page_size=5).delisted()["data"]) == 5
    assert len(responses.calls) == 2


@responses.activate
@pytest.mark.django_db
def test_failed_stat_is_a_schema_change(load_fixture: Callable[[str], Any]) -> None:
    responses.post(URL, json={**load_fixture("tpex/delisted.json"), "stat": "error"})
    with pytest.raises(SchemaChangedError, match="stat"):
        _client().delisted()


@responses.activate
@pytest.mark.django_db
def test_missing_table_is_a_schema_change() -> None:
    responses.post(URL, json={"stat": "ok", "tables": []})
    with pytest.raises(SchemaChangedError, match="table"):
        _client().delisted()


@responses.activate
def test_refused_request_is_reported() -> None:
    responses.post(URL, status=403)
    with pytest.raises(SourceRequestError, match="TPEx"):
        _client().delisted()


def test_interval_below_three_seconds_is_refused() -> None:
    fast = RateLimit(
        min_interval=1.0, max_retries=0, backoff_base=1.0, backoff_max=1.0, cooldown=1.0
    )
    with pytest.raises(ImproperlyConfigured, match="3 seconds"):
        TpexClient(RateLimiter(fast), "https://www.tpex.org.tw", 30.0)


@responses.activate
@pytest.mark.django_db
def test_missing_total_count_is_a_schema_change(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture("tpex/delisted.json")
    table = {k: v for k, v in payload["tables"][0].items() if k != "totalCount"}
    responses.post(URL, json={**payload, "tables": [table]})
    with pytest.raises(SchemaChangedError, match="totalCount"):
        _client().delisted()
