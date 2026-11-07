import json
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import responses
import time_machine
from django.conf import settings
from responses import matchers

from apps.etl.adapters.mops import COMPANY_REVENUE_PATH, MopsAdapter
from apps.etl.company_revenue import company_months, parse_company_revenue
from apps.etl.errors import SchemaChangedError
from apps.etl.models import DataQualityLog, EtlRun, RawMops
from apps.etl.mops_history import MISMATCH_CHECKS, REVENUE_COMPANY
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, MonthlyRevenue, StockMaster

TAIPEI = ZoneInfo("Asia/Taipei")
TODAY = datetime(2026, 11, 8, 10, 0, tzinfo=TAIPEI)
LIMIT = RateLimit(min_interval=6.0, max_retries=2, backoff_base=1.0, backoff_max=2.0, cooldown=5.0)
API = f"{settings.MOPS_API_BASE_URL}{COMPANY_REVENUE_PATH}"
SPAN = (date(2013, 1, 1), date(2014, 1, 31))
FOREIGN = "功能性貨幣(人民幣)"


@pytest.fixture
def answer(fixture_path: Callable[[str], Path]) -> Callable[[str], dict[str, Any]]:
    folder = "mops/company_revenue/t05st10_ifrs_{}.json"
    return lambda name: json.loads(fixture_path(folder.format(name)).read_text("utf-8"))


def test_a_month_reads_this_month_and_the_same_month_last_year_in_twd(
    answer: Callable[[str], dict[str, Any]],
) -> None:
    assert parse_company_revenue(answer("5349_106_6"), "2017-06") == (595_274_000, 744_139_000)


def test_a_foreign_issuer_reads_the_twd_column_not_its_own_currency(
    answer: Callable[[str], dict[str, Any]],
) -> None:
    assert parse_company_revenue(answer("1262_104_6"), "2015-06") == (542_727_000, 523_405_000)


def test_a_month_mops_holds_no_figures_for_reads_as_none(
    answer: Callable[[str], dict[str, Any]],
) -> None:
    assert parse_company_revenue(answer("5349_102_1"), "2013-01") is None


@pytest.mark.parametrize(
    "broken",
    [
        lambda payload: {**payload, "code": 500, "message": "查詢過量"},
        lambda payload: {**payload, "result": {**payload["result"], "yymm": "10605"}},
        lambda payload: {
            **payload,
            "result": {
                **payload["result"],
                "titles": [
                    payload["result"]["titles"][0],
                    {"main": "合併營業收入淨額", "sub": [{"main": FOREIGN, "sub": []}]},
                ],
            },
        },
        lambda payload: {**payload, "result": {**payload["result"], "data": [["本月", "─"]]}},
    ],
    ids=["failed", "another-month", "no-twd-column", "no-amount"],
)
def test_an_answer_that_is_not_the_month_asked_in_twd_is_refused(
    answer: Callable[[str], dict[str, Any]], broken: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    with pytest.raises(SchemaChangedError):
        parse_company_revenue(broken(answer("1262_104_6")), "2015-06")


def listed(code: str, list_date: date | None, delist_date: date | None, **extra: Any) -> None:
    StockMaster.objects.create(
        stock_id=code,
        name=code,
        market="TPEx",
        list_date=list_date,
        delist_date=delist_date,
        **extra,
    )


@pytest.mark.django_db
def test_months_run_from_the_listing_month_to_the_delisting_month_less_stored_ones() -> None:
    listed("5349", None, date(2014, 3, 15))
    DailyPrice.objects.create(
        stock_id="5349", date=date(2013, 11, 20), close=10, volume=1_000, turnover=10_000
    )
    MonthlyRevenue.objects.create(
        stock_id="5349", period="2013-12", revenue=1, announce_date=date(2014, 1, 13)
    )
    months = company_months("5349", date(2013, 1, 1), date(2026, 12, 31), TODAY.date(), 20)
    assert months == ["2013-11", "2014-01", "2014-02", "2014-03"]


@pytest.mark.django_db
def test_a_listed_stock_stops_at_the_months_the_daily_scan_still_watches() -> None:
    listed("2330", date(1994, 9, 5), None)
    months = company_months("2330", date(2026, 7, 1), date(2026, 12, 31), TODAY.date(), 20)
    assert months == ["2026-07", "2026-08", "2026-09"]


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("code", "extra"),
    [("9105", {}), ("0050", {}), ("2905", {"industry": "存託憑證"})],
    ids=["depositary-prefix", "etf", "depositary-industry"],
)
def test_only_common_stocks_are_asked_for(code: str, extra: dict[str, str]) -> None:
    listed(code, date(2010, 1, 4), None, **extra)
    assert company_months(code, date(2014, 1, 1), date(2014, 12, 31), TODAY.date(), 20) == []


@pytest.fixture
def adapter(fake_sleep: Callable[[float], None]) -> MopsAdapter:
    days = [date(2012, 1, 1) + timedelta(days=n) for n in range(15 * 366)]
    calendar = TradingDays([d for d in days if d.weekday() < 5], days[0], days[-1])
    return MopsAdapter(calendar, RateLimiter(LIMIT, sleep=fake_sleep))


@pytest.fixture
def delisted() -> None:
    listed("5349", date(2010, 1, 4), date(2020, 11, 4))
    MonthlyRevenue.objects.bulk_create(
        MonthlyRevenue(
            stock_id="5349",
            period=f"2013-{month:02d}",
            revenue=700_000_000,
            announce_date=date(2013, month + 1, 11) if month < 12 else date(2014, 1, 13),
        )
        for month in range(2, 13)
    )


def serve(answer: Callable[[str], dict[str, Any]]) -> None:
    for name, year, month in (("5349_102_1", "102", "1"), ("5349_103_1", "103", "1")):
        body = {
            "companyId": "5349",
            "dataType": "2",
            "year": year,
            "month": month,
            "subsidiaryCompanyId": "",
        }
        responses.post(API, json=answer(name), match=[matchers.json_params_matcher(body)])


def backfill(adapter: MopsAdapter) -> EtlRun:
    with time_machine.travel(TODAY, tick=False):
        return adapter.backfill(REVENUE_COMPANY, ["5349"], *SPAN)


@pytest.mark.django_db
@responses.activate
@pytest.mark.usefixtures("delisted")
def test_a_missing_month_is_added_at_the_conservative_date_and_stored_ones_stay(
    adapter: MopsAdapter, answer: Callable[[str], dict[str, Any]]
) -> None:
    serve(answer)
    assert backfill(adapter).rows == 1
    added = MonthlyRevenue.objects.get(stock_id="5349", period="2014-01")
    assert (added.revenue, added.announce_date) == (644_212_000, date(2014, 2, 11))
    assert added.yoy == pytest.approx(644_212 / 702_027 - 1)
    assert added.mom == pytest.approx(644_212_000 / 700_000_000 - 1)
    assert not MonthlyRevenue.objects.filter(period="2013-01").exists()
    assert MonthlyRevenue.objects.get(period="2013-12").revenue == 700_000_000
    assert RawMops.objects.filter(dataset=REVENUE_COMPANY).count() == 2
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[REVENUE_COMPANY])
    span = ["2013-01-01", "2014-01-31"]
    assert (logged.detail["stock_id"], logged.detail["span"]) == ("5349", span)
    assert logged.detail["added"] == [{"stock_id": "5349", "period": "2014-01"}]
    assert logged.detail["absent"] == ["2013-01"]


@pytest.mark.django_db
@responses.activate
@pytest.mark.usefixtures("delisted")
def test_a_rerun_asks_only_for_the_months_still_missing(
    adapter: MopsAdapter, answer: Callable[[str], dict[str, Any]]
) -> None:
    serve(answer)
    backfill(adapter)
    assert backfill(adapter).rows == 0
    asked = [json.loads(call.request.body or "{}")["year"] for call in responses.calls]
    assert asked == ["102", "103", "102"]
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[REVENUE_COMPANY])
    assert logged.detail["added"] == [{"stock_id": "5349", "period": "2014-01"}]


def test_the_whole_market_cannot_be_asked_at_once(adapter: MopsAdapter) -> None:
    with pytest.raises(ValueError, match="stock by stock"):
        adapter.run(REVENUE_COMPANY, *SPAN)
