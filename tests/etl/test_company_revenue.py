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
from apps.etl.mops_history import (
    MISMATCH_CHECKS,
    REVENUE_COMPANY,
    REVENUE_HISTORY,
    STATEMENT_HISTORY,
    STATEMENT_VIEWER,
)
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, MonthlyRevenue, StockMaster

TAIPEI = ZoneInfo("Asia/Taipei")
TODAY = datetime(2026, 11, 8, 10, 0, tzinfo=TAIPEI)
LIMIT = RateLimit(min_interval=6.0, max_retries=2, backoff_base=1.0, backoff_max=2.0, cooldown=5.0)
API = f"{settings.MOPS_API_BASE_URL}{COMPANY_REVENUE_PATH}"
SPAN = (date(2013, 1, 1), date(2026, 11, 7))
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


def test_a_company_mops_finds_no_match_for_that_month_reads_as_none(
    answer: Callable[[str], dict[str, Any]],
) -> None:
    assert parse_company_revenue(answer("1258_112_5"), "2023-05") is None


@pytest.mark.parametrize(
    "broken",
    [
        lambda payload: {**payload, "code": 500, "message": "查詢過量"},
        lambda payload: {"code": 406, "result": None, "message": "參數錯誤"},
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
    ids=["failed", "another-406", "another-month", "no-twd-column", "no-amount"],
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
def test_months_run_from_2014_or_the_listing_month_to_the_delisting_month_less_stored_ones() -> (
    None
):
    listed("5349", None, date(2014, 3, 15))
    DailyPrice.objects.create(
        stock_id="5349", date=date(2013, 11, 20), close=10, volume=1_000, turnover=10_000
    )
    MonthlyRevenue.objects.create(
        stock_id="5349", period="2014-02", revenue=1, announce_date=date(2014, 3, 11)
    )
    months = company_months("5349", date(2013, 1, 1), date(2026, 12, 31), TODAY.date(), 20, None)
    assert months == ["2014-01", "2014-03"]


@pytest.mark.django_db
def test_a_listed_stock_stops_at_the_months_the_daily_scan_still_watches() -> None:
    listed("2330", date(1994, 9, 5), None)
    months = company_months("2330", date(2026, 7, 1), date(2026, 12, 31), TODAY.date(), 20, None)
    assert months == ["2026-07", "2026-08", "2026-09"]


@pytest.mark.django_db
def test_months_the_daily_scan_has_dated_are_never_asked_for(adapter: MopsAdapter) -> None:
    listed("2330", date(1994, 9, 5), None)
    day = date(2026, 11, 5)
    EtlRun.objects.create(
        source="mops", dataset="revenue_scan", start_date=day, end_date=day, status="success"
    )
    scanned = adapter.scanned_from(REVENUE_COMPANY)
    later = date(2026, 12, 10)
    months = company_months("2330", date(2026, 7, 1), later, later, 20, scanned)
    assert (scanned, months) == ("2026-10", ["2026-07", "2026-08", "2026-09"])


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("code", "extra"),
    [("9105", {}), ("0050", {}), ("2905", {"industry": "存託憑證"})],
    ids=["depositary-prefix", "etf", "depositary-industry"],
)
def test_only_common_stocks_are_asked_for(code: str, extra: dict[str, str]) -> None:
    listed(code, date(2010, 1, 4), None, **extra)
    months = company_months(code, date(2014, 1, 1), date(2014, 12, 31), TODAY.date(), 20, None)
    assert months == []


@pytest.fixture
def adapter(fake_sleep: Callable[[float], None]) -> MopsAdapter:
    days = [date(2012, 1, 1) + timedelta(days=n) for n in range(15 * 366)]
    calendar = TradingDays([d for d in days if d.weekday() < 5], days[0], days[-1])
    return MopsAdapter(calendar, RateLimiter(LIMIT, sleep=fake_sleep))


@pytest.fixture
def delisted() -> None:
    listed("5349", date(2010, 1, 4), date(2014, 3, 10))
    MonthlyRevenue.objects.create(
        stock_id="5349", period="2013-12", revenue=700_000_000, announce_date=date(2014, 1, 13)
    )


def month_of(
    payload: dict[str, Any], yymm: str, amount: str | None = None, asked_at: str | None = None
) -> dict[str, Any]:
    """A recorded answer moved to another month, with this month's amount or query time changed."""
    result = {**payload["result"], "yymm": yymm}
    if amount is not None:
        result["data"] = [["本月", amount], *payload["result"]["data"][1:]]
    return {**payload, "result": result, "datetime": asked_at or payload["datetime"]}


def answers(answer: Callable[[str], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    january = answer("5349_103_1")
    return {
        "1": january,
        "2": month_of(january, "10302", "600,000"),
        "3": month_of(answer("5349_102_1"), "10303"),
    }


def serve(replies: dict[str, Callable[[], dict[str, Any]]]) -> None:
    for month, reply in replies.items():
        body = {
            "companyId": "5349",
            "dataType": "2",
            "year": "103",
            "month": month,
            "subsidiaryCompanyId": "",
        }
        responses.add_callback(
            responses.POST,
            API,
            callback=replying(reply),
            content_type="application/json",
            match=[matchers.json_params_matcher(body)],
        )


def replying(reply: Callable[[], dict[str, Any]]) -> Callable[[Any], tuple[int, dict, str]]:
    def callback(_request: Any) -> tuple[int, dict, str]:
        return 200, {}, json.dumps(reply())

    return callback


def fixed(payload: dict[str, Any]) -> Callable[[], dict[str, Any]]:
    return lambda: payload


def backfill(adapter: MopsAdapter) -> EtlRun:
    with time_machine.travel(TODAY, tick=False):
        return adapter.backfill(REVENUE_COMPANY, ["5349"], *SPAN)


@pytest.mark.django_db
@responses.activate
@pytest.mark.usefixtures("delisted")
def test_missing_months_are_added_at_the_conservative_date_from_mops_figures_only(
    adapter: MopsAdapter, answer: Callable[[str], dict[str, Any]]
) -> None:
    serve({month: fixed(reply) for month, reply in answers(answer).items()})
    assert backfill(adapter).rows == 2
    added = {row.period: row for row in MonthlyRevenue.objects.filter(announce_date__year=2014)}
    january, february = added["2014-01"], added["2014-02"]
    assert (january.revenue, january.announce_date) == (644_212_000, date(2014, 2, 11))
    assert january.yoy == pytest.approx(644_212 / 702_027 - 1)
    assert january.mom is None
    assert february.mom == pytest.approx(600_000 / 644_212 - 1)
    assert not MonthlyRevenue.objects.filter(period="2014-03").exists()
    assert MonthlyRevenue.objects.get(period="2013-12").revenue == 700_000_000
    assert RawMops.objects.filter(dataset=REVENUE_COMPANY).count() == 3
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[REVENUE_COMPANY])
    assert (logged.detail["stock_id"], logged.detail["span"]) == (
        "5349",
        ["2013-01-01", "2014-03-31"],
    )
    assert logged.detail["added"] == [
        {"stock_id": "5349", "period": "2014-01"},
        {"stock_id": "5349", "period": "2014-02"},
    ]
    assert logged.detail["absent"] == ["2014-03"]


@pytest.mark.django_db
@responses.activate
@pytest.mark.usefixtures("delisted")
def test_a_rerun_asks_only_for_the_months_still_missing_and_keeps_one_raw_per_answer(
    adapter: MopsAdapter, answer: Callable[[str], dict[str, Any]]
) -> None:
    replies = answers(answer)
    times = iter(["115/11/08 05:51:50", "115/11/09 08:00:00"])
    serve(
        {
            "1": fixed(replies["1"]),
            "2": fixed(replies["2"]),
            "3": lambda: {**replies["3"], "datetime": next(times)},
        }
    )
    backfill(adapter)
    assert backfill(adapter).rows == 0
    asked = [json.loads(call.request.body or "{}")["month"] for call in responses.calls]
    assert asked == ["1", "2", "3", "3"]
    assert RawMops.objects.filter(dataset=REVENUE_COMPANY).count() == 3
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[REVENUE_COMPANY])
    assert [entry["period"] for entry in logged.detail["added"]] == ["2014-01", "2014-02"]


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("scan", "dataset", "first"),
    [
        ("revenue_scan", REVENUE_HISTORY, "2026-10"),
        ("statement_scan", STATEMENT_VIEWER, "2026-Q3"),
        ("statement_scan", STATEMENT_HISTORY, "2026-Q3"),
    ],
)
def test_the_first_period_a_scan_dated_bounds_every_fill(
    adapter: MopsAdapter, scan: str, dataset: str, first: str
) -> None:
    for day, status in ((date(2026, 9, 1), "failed"), (date(2026, 11, 5), "success")):
        EtlRun.objects.create(
            source="mops", dataset=scan, start_date=day, end_date=day, status=status
        )
    assert adapter.scanned_from(dataset) == first


@pytest.mark.django_db
def test_the_summary_history_refuses_months_the_daily_scan_dated(adapter: MopsAdapter) -> None:
    day = date(2026, 11, 5)
    EtlRun.objects.create(
        source="mops", dataset="revenue_scan", start_date=day, end_date=day, status="success"
    )
    with (
        time_machine.travel(datetime(2027, 1, 20, 10, 0, tzinfo=TAIPEI), tick=False),
        pytest.raises(ValueError, match="daily scan dated"),
    ):
        adapter.run(REVENUE_HISTORY, date(2026, 9, 1), date(2026, 11, 30))


def test_the_whole_market_cannot_be_asked_at_once(adapter: MopsAdapter) -> None:
    with pytest.raises(ValueError, match="stock by stock"):
        adapter.run(REVENUE_COMPANY, *SPAN)
