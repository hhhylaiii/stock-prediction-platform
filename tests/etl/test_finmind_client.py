from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import responses
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from responses import matchers

from apps.etl.adapters import limiter_for
from apps.etl.adapters.base import SchemaChangedError, SourceRequestError
from apps.etl.adapters.finmind import FinmindClient, maintenance_wait
from apps.etl.models import RawFinmind
from apps.etl.ratelimit import RateLimit, RateLimiter, ThrottledError
from tests.etl.test_ratelimit import FakeClock

URL = "https://api.finmindtrade.com/api/v4/data"
TAIPEI = ZoneInfo("Asia/Taipei")
SATURDAY_NOON = datetime(2026, 10, 31, 12, 0, tzinfo=TAIPEI)
FAKE_KEY = "fake-finmind-key"


def _limit(min_interval: float = 6.0, cooldown: float = 3700.0) -> RateLimit:
    return RateLimit(
        min_interval=min_interval,
        max_retries=2,
        backoff_base=1.0,
        backoff_max=2.0,
        cooldown=cooldown,
    )


def _client(clock: FakeClock, key: str = FAKE_KEY, limit: RateLimit | None = None) -> FinmindClient:
    limiter = RateLimiter(limit or _limit(), clock=clock, sleep=clock.sleep)
    return FinmindClient(
        limiter,
        "https://api.finmindtrade.com",
        30.0,
        key,
        now=lambda: SATURDAY_NOON,
        sleep=clock.sleep,
    )


def _ok(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"msg": "success", "status": 200, "data": rows}


@pytest.mark.parametrize(
    ("key", "interval", "allowed"),
    [(FAKE_KEY, 6.0, True), (FAKE_KEY, 5.9, False), ("", 12.0, True), ("", 6.0, False)],
)
def test_interval_must_fit_the_hourly_quota(key: str, interval: float, allowed: bool) -> None:
    limiter = RateLimiter(_limit(interval))
    if allowed:
        FinmindClient(limiter, "https://api.finmindtrade.com", 30.0, key)
        return
    with pytest.raises(ImproperlyConfigured, match="per hour"):
        FinmindClient(limiter, "https://api.finmindtrade.com", 30.0, key)


def test_configured_client_fits_its_quota() -> None:
    client = FinmindClient.build(limiter_for("finmind"))
    assert client.token == settings.FINMIND_API_TOKEN == ""


@responses.activate
@pytest.mark.django_db
def test_no_rolling_hour_exceeds_the_quota() -> None:
    clock = FakeClock()
    sent: list[float] = []

    def record(request: Any) -> tuple[int, dict[str, str], str]:
        sent.append(clock.now)
        return 200, {}, '{"msg": "success", "status": 200, "data": []}'

    responses.add_callback(responses.GET, URL, callback=record)
    client = _client(clock)
    for _ in range(1300):
        client.data("trading_date", "TaiwanStockTradingDate")
    busiest = max(sum(1 for t in sent if start <= t < start + 3600) for start in sent)
    assert busiest <= 600
    assert len(sent) == 1300


@responses.activate
@pytest.mark.django_db
def test_rows_are_returned_and_the_response_is_kept_raw() -> None:
    rows = [{"date": "2010-01-04"}, {"date": "2010-01-05"}]
    responses.get(
        URL,
        json=_ok(rows),
        match=[
            matchers.query_param_matcher(
                {
                    "dataset": "TaiwanStockTradingDate",
                    "start_date": "2010-01-01",
                    "end_date": "2010-01-05",
                }
            ),
            matchers.header_matcher({"Authorization": f"Bearer {FAKE_KEY}"}),
        ],
    )
    client = _client(FakeClock())
    got = client.data(
        "trading_date", "TaiwanStockTradingDate", start_date="2010-01-01", end_date="2010-01-05"
    )
    assert got == rows
    stored = RawFinmind.objects.get()
    assert stored.dataset == "trading_date"
    assert stored.request_params == {
        "dataset": "TaiwanStockTradingDate",
        "start_date": "2010-01-01",
        "end_date": "2010-01-05",
    }


@responses.activate
@pytest.mark.django_db
def test_anonymous_requests_send_no_authorization() -> None:
    responses.get(URL, json=_ok([]))
    clock = FakeClock()
    _client(clock, key="", limit=_limit(12.0)).data("stock_info", "TaiwanStockInfo")
    assert "Authorization" not in responses.calls[0].request.headers


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    "throttled",
    [
        {"status": 402},
        {"status": 200, "json": {"msg": "Requests reach the upper limit.", "status": 402}},
    ],
    ids=["http-402", "body-402"],
)
def test_quota_exhaustion_cools_down_then_retries(throttled: dict[str, Any]) -> None:
    responses.get(URL, **throttled)
    responses.get(URL, json=_ok([{"date": "2010-01-04"}]))
    clock = FakeClock()
    assert _client(clock).data("trading_date", "TaiwanStockTradingDate") == [{"date": "2010-01-04"}]
    assert 3700.0 in clock.sleeps


@responses.activate
def test_quota_exhaustion_that_persists_is_raised() -> None:
    responses.get(URL, status=402)
    with pytest.raises(ThrottledError):
        _client(FakeClock()).data("trading_date", "TaiwanStockTradingDate")
    assert len(responses.calls) == 3


@responses.activate
def test_error_status_in_the_body_is_a_refused_request() -> None:
    responses.get(URL, json={"msg": "dataset not found", "status": 400})
    with pytest.raises(SourceRequestError, match="dataset not found"):
        _client(FakeClock()).data("stock_info", "TaiwanStockInfoX")


@responses.activate
@pytest.mark.django_db
def test_data_that_is_not_a_list_is_a_schema_change() -> None:
    responses.get(URL, json={"msg": "success", "status": 200, "data": {"date": "2010-01-04"}})
    with pytest.raises(SchemaChangedError, match="list"):
        _client(FakeClock()).data("trading_date", "TaiwanStockTradingDate")
    assert RawFinmind.objects.filter(dataset="trading_date").count() == 1


@pytest.mark.parametrize(
    ("moment", "wait"),
    [
        (datetime(2026, 11, 1, 3, 0, tzinfo=TAIPEI), 4 * 3600.0),
        (datetime(2026, 11, 1, 0, 0, tzinfo=TAIPEI), 7 * 3600.0),
        (datetime(2026, 11, 1, 6, 59, 30, tzinfo=TAIPEI), 30.0),
        (datetime(2026, 11, 1, 7, 0, tzinfo=TAIPEI), 0.0),
        (datetime(2026, 10, 31, 23, 59, tzinfo=TAIPEI), 0.0),
        (datetime(2026, 11, 2, 3, 0, tzinfo=TAIPEI), 0.0),
    ],
)
def test_maintenance_window_is_sunday_midnight_to_seven(moment: datetime, wait: float) -> None:
    assert maintenance_wait(moment) == pytest.approx(wait)


@responses.activate
@pytest.mark.django_db
def test_requests_wait_out_the_maintenance_window() -> None:
    responses.get(URL, json=_ok([]))
    clock = FakeClock()
    limiter = RateLimiter(_limit(), clock=clock, sleep=clock.sleep)
    client = FinmindClient(
        limiter,
        "https://api.finmindtrade.com",
        30.0,
        FAKE_KEY,
        now=lambda: datetime(2026, 11, 1, 5, 0, tzinfo=TAIPEI),
        sleep=clock.sleep,
    )
    client.data("stock_info", "TaiwanStockInfo")
    assert clock.sleeps[0] == pytest.approx(2 * 3600.0)


@responses.activate
@pytest.mark.django_db
def test_retries_after_a_cooldown_also_wait_out_maintenance() -> None:
    responses.get(URL, status=402)
    responses.get(URL, json=_ok([]))
    clock = FakeClock()
    late_saturday = datetime(2026, 10, 31, 23, 50, tzinfo=TAIPEI)
    limiter = RateLimiter(_limit(), clock=clock, sleep=clock.sleep)
    client = FinmindClient(
        limiter,
        "https://api.finmindtrade.com",
        30.0,
        FAKE_KEY,
        now=lambda: late_saturday + timedelta(seconds=clock.now - 1000.0),
        sleep=clock.sleep,
    )
    client.data("stock_info", "TaiwanStockInfo")
    assert len(responses.calls) == 2
    assert clock.sleeps[0] == pytest.approx(3700.0)
    assert clock.sleeps[1] == pytest.approx(6 * 3600 + 8 * 60 + 20)
