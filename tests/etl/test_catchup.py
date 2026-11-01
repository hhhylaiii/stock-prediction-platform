from collections.abc import Callable
from datetime import date, datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import pytest
import responses
import time_machine
from django.core.management import call_command
from django.core.management.base import CommandError
from responses import matchers

from apps.etl.adapters import ADAPTERS
from apps.etl.catchup import (
    JOBS,
    Job,
    catch_up,
    covered_days,
    missing_days,
    rows_present,
    select_jobs,
)
from apps.etl.models import DataQualityLog, EtlRun, RawTwse
from apps.market.calendar import TradingDays
from apps.market.models import (
    DailyChip,
    DailyPrice,
    IndexDaily,
    SharesOutstanding,
    StockMaster,
)

MI_INDEX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
MI_QFIIS = "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"
PRICES = "twse/daily_price/mi_index_20261028.json"
SHARES = "twse/shares_outstanding/mi_qfiis_20261028.json"
PRICE_JOB = Job("twse", "daily_price", rows_present(DailyPrice, "TWSE"))
SHARE_JOB = Job("twse", "shares_outstanding", rebuild_later=True)
DAYS = [date(2026, 10, 27), date(2026, 10, 28), date(2026, 10, 29)]

pytestmark = pytest.mark.usefixtures("october_2026_in_db", "quiet_limiter", "listed")


@pytest.fixture
def quiet_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)


@pytest.fixture
def listed(db: None) -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code, name=code, market="TWSE")
        for code in ("0050", "1101", "2317", "2330", "2368")
    )


def _serve(url: str, payload: dict[str, Any], day: date, **params: str) -> None:
    query = {"date": f"{day:%Y%m%d}", **params, "response": "json"}
    responses.get(
        url,
        json={**payload, "date": f"{day:%Y%m%d}"},
        match=[matchers.query_param_matcher(query)],
    )


def _serve_prices(load_fixture: Callable[[str], Any], day: date) -> None:
    _serve(MI_INDEX, load_fixture(PRICES), day, type="ALLBUT0999")


def _run(source: str, dataset: str, start: date, end: date, **fields: Any) -> EtlRun:
    return EtlRun.objects.create(
        source=source, dataset=dataset, start_date=start, end_date=end, **fields
    )


def test_a_day_counts_as_done_when_a_run_covered_it() -> None:
    _run("twse", "shares_outstanding", DAYS[0], DAYS[1], status="success")
    _run(
        "twse",
        "shares_outstanding",
        DAYS[2],
        date(2026, 10, 30),
        status="failed",
        checkpoint={"last_date": "2026-10-29"},
    )
    _run("twse", "shares_outstanding", date(2026, 10, 30), date(2026, 10, 30), status="running")
    asked = [*DAYS, date(2026, 10, 30)]
    assert covered_days("twse", "shares_outstanding", asked) == set(DAYS)
    assert covered_days("tpex", "shares_outstanding", asked) == set()


def test_a_day_with_rows_of_the_market_is_done_without_any_run() -> None:
    DailyPrice.objects.create(stock_id="2330", date=DAYS[0], volume=1, turnover=1)
    DailyPrice.objects.create(stock_id="6488", date=DAYS[1], volume=1, turnover=1)
    assert missing_days(PRICE_JOB, DAYS) == DAYS[1:]


def test_a_chip_day_is_done_only_for_the_dataset_whose_columns_are_filled() -> None:
    DailyChip.objects.create(stock_id="2330", date=DAYS[0], margin_balance=1)
    jobs = {job.dataset: job for job in select_jobs("twse")}
    assert missing_days(jobs["margin"], DAYS) == DAYS[1:]
    assert missing_days(jobs["institutional"], DAYS) == DAYS


def test_an_index_day_is_done_when_the_market_index_has_a_row() -> None:
    IndexDaily.objects.create(index_id="TPEX", date=DAYS[0], close=1)
    IndexDaily.objects.create(index_id="TAIEX", date=DAYS[1], close=1)
    jobs = {job.key: job for job in JOBS}
    assert missing_days(jobs["tpex/index_daily"], DAYS) == DAYS[1:]
    assert missing_days(jobs["twse/index_daily"], DAYS) == [DAYS[0], DAYS[2]]


def test_every_daily_exchange_dataset_is_caught_up_and_share_counts_are_rebuilt() -> None:
    keys = {job.key: job for job in JOBS}
    for source in ("twse", "tpex"):
        datasets = {job.dataset for job in JOBS if job.source == source}
        assert datasets == set(ADAPTERS[source].datasets)
        assert keys[f"{source}/shares_outstanding"].rebuild_later
        assert keys[f"{source}/shares_outstanding"].present is None
    assert {job.dataset for job in JOBS if job.source == "yahoo"} == {"sox", "tsm_adr", "usdtwd"}
    assert [job.key for job in JOBS if job.rebuild_later] == [
        "twse/shares_outstanding",
        "tpex/shares_outstanding",
    ]


@responses.activate
def test_missing_days_are_fetched_in_order_and_checked_again(
    october_2026_in_db: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    DailyPrice.objects.create(stock_id="2330", date=DAYS[0], volume=1, turnover=1)
    for day in DAYS[1:]:
        _serve_prices(load_fixture, day)
    report = catch_up(DAYS[0], DAYS[-1], october_2026_in_db, [PRICE_JOB])
    assert report.written == {"twse/daily_price": ["2026-10-28", "2026-10-29"]}
    asked = [parse_qs(urlsplit(str(call.request.url)).query)["date"] for call in responses.calls]
    assert asked == [["20261028"], ["20261029"]]
    assert set(DataQualityLog.objects.values_list("date", flat=True)) == set(DAYS[1:])
    assert report.lines()[-1] == "checked quality again: 2026-10-28, 2026-10-29"


@responses.activate
def test_a_day_not_published_yet_is_reported_and_others_go_on(
    october_2026_in_db: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    _serve_prices(load_fixture, DAYS[0])
    _serve(
        MI_INDEX,
        load_fixture("twse/daily_price/mi_index_20261025_no_data.json"),
        DAYS[1],
        type="ALLBUT0999",
    )
    _serve_prices(load_fixture, DAYS[2])
    report = catch_up(DAYS[0], DAYS[-1], october_2026_in_db, [PRICE_JOB])
    assert report.written == {"twse/daily_price": ["2026-10-27", "2026-10-29"]}
    assert list(report.not_ready) == ["twse/daily_price 2026-10-28"]
    assert report.failed == {}


@responses.activate
def test_a_broken_answer_is_a_failure_that_does_not_stop_other_jobs(
    october_2026_in_db: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    responses.get(MI_INDEX, json={"stat": "OK", "date": "20261027", "tables": [{"x": 1}]})
    _serve(MI_QFIIS, load_fixture(SHARES), DAYS[0], selectType="ALLBUT0999")
    report = catch_up(DAYS[0], DAYS[0], october_2026_in_db, [PRICE_JOB, SHARE_JOB])
    assert list(report.failed) == ["twse/daily_price 2026-10-27"]
    assert report.written == {"twse/shares_outstanding": ["2026-10-27"]}


@responses.activate
def test_a_job_stops_at_its_first_failure(october_2026_in_db: TradingDays) -> None:
    responses.get(MI_INDEX, json={"stat": "OK", "date": "20261027", "tables": [{"x": 1}]})
    report = catch_up(DAYS[0], DAYS[-1], october_2026_in_db, [PRICE_JOB])
    assert list(report.failed) == ["twse/daily_price 2026-10-27"]
    assert len(responses.calls) == 1
    assert not DataQualityLog.objects.exists()


@responses.activate
def test_shares_caught_up_late_rebuild_the_later_days_from_raw(
    october_2026_in_db: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(SHARES)
    for day in (DAYS[0], DAYS[2]):
        _serve(MI_QFIIS, payload, day, selectType="ALLBUT0999")
        call_command("fetch", source="twse", dataset="shares_outstanding", date=day.isoformat())
    original = dict(
        SharesOutstanding.objects.filter(stock_id="2330").values_list("effective_date", "shares")
    )
    assert list(original) == [DAYS[0]]
    changed = {
        **payload,
        "data": [
            [*row[:3], "99,000", *row[4:]] if row[0] == "2330" else row for row in payload["data"]
        ],
    }
    _serve(MI_QFIIS, changed, DAYS[1], selectType="ALLBUT0999")
    report = catch_up(DAYS[0], DAYS[2], october_2026_in_db, [SHARE_JOB])
    assert report.written == {"twse/shares_outstanding": ["2026-10-28"]}
    assert report.rebuilt == {"twse/shares_outstanding": ["2026-10-29"]}
    counts = dict(
        SharesOutstanding.objects.filter(stock_id="2330").values_list("effective_date", "shares")
    )
    assert counts == {DAYS[0]: original[DAYS[0]], DAYS[1]: 99000, DAYS[2]: original[DAYS[0]]}


@responses.activate
def test_a_later_day_without_raw_is_reported_as_a_failed_rebuild(
    october_2026_in_db: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(SHARES)
    for day in DAYS:
        _serve(MI_QFIIS, payload, day, selectType="ALLBUT0999")
    call_command("fetch", source="twse", dataset="shares_outstanding", date="2026-10-29")
    RawTwse.objects.all().delete()
    report = catch_up(DAYS[0], DAYS[1], october_2026_in_db, [SHARE_JOB])
    assert report.written == {"twse/shares_outstanding": ["2026-10-27", "2026-10-28"]}
    assert report.rebuilt == {}
    assert list(report.failed) == ["twse/shares_outstanding rebuild 2026-10-28..2026-10-29"]


def test_catchup_command_prints_the_report(capsys: pytest.CaptureFixture[str]) -> None:
    DailyPrice.objects.create(stock_id="2330", date=DAYS[0], volume=1, turnover=1)
    call_command(
        "catchup", since="2026-10-27", until="2026-10-27", source="twse", dataset="daily_price"
    )
    assert capsys.readouterr().out.strip() == "nothing to catch up"


@responses.activate
def test_catchup_command_fails_when_a_step_failed(capsys: pytest.CaptureFixture[str]) -> None:
    responses.get(MI_INDEX, json={"stat": "OK", "date": "20261027", "tables": [{"x": 1}]})
    with pytest.raises(CommandError, match="1 catch-up step"):
        call_command(
            "catchup", since="2026-10-27", until="2026-10-27", source="twse", dataset="daily_price"
        )
    assert "failed twse/daily_price 2026-10-27" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"since": "2026-10-29", "until": "2026-10-27"}, "must not be after"),
        ({"since": "2026-10-27", "source": "finmind"}, "no catch-up job"),
        ({"since": "2026-10-27", "until": "2026-10-30"}, "after today"),
    ],
)
@time_machine.travel(datetime(2026, 10, 29, 20, tzinfo=ZoneInfo("Asia/Taipei")), tick=False)
def test_catchup_command_rejects_bad_options(options: dict[str, str], message: str) -> None:
    with pytest.raises(CommandError, match=message):
        call_command("catchup", **options)
