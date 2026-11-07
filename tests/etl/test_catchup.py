import logging
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
from apps.etl.adapters.twse import TwseAdapter
from apps.etl.catchup import (
    JOBS,
    Job,
    catch_up,
    covered_days,
    missing_days,
    plan_catch_up,
    rows_present,
    select_jobs,
)
from apps.etl.models import DataQualityLog, EtlRun, RawTwse
from apps.etl.raw import store_raw
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
TWTB4U = "https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U"
DAY_TRADES = "twse/day_trade/twtb4u_20261028.json"
PRICES = "twse/daily_price/mi_index_20261028.json"
SHARES = "twse/shares_outstanding/mi_qfiis_20261028.json"
PRICE_JOB = Job("twse", "daily_price", rows_present(DailyPrice, "TWSE"))
SHARE_JOB = Job("twse", "shares_outstanding", rebuild_later=True)
DAYS = [date(2026, 10, 27), date(2026, 10, 28), date(2026, 10, 29)]
EVENING = datetime(2026, 10, 29, 20, tzinfo=ZoneInfo("Asia/Taipei"))

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


def _quoted(load_fixture: Callable[[str], Any], day: date) -> None:
    """A day whose TWSE quotes are stored and written, as a daily run leaves it."""
    params = {"date": f"{day:%Y%m%d}", "type": "ALLBUT0999", "response": "json"}
    store_raw(RawTwse, "daily_price", params, {**load_fixture(PRICES), "date": params["date"]})
    DailyPrice.objects.create(stock_id="2330", date=day, volume=1, turnover=1)


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
def test_two_missed_nights_caught_up_leave_the_final_day_trades_of_t_minus_2(
    october_2026_in_db: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    redone = [date(2026, 10, 22), *DAYS]
    for day in redone:
        _quoted(load_fixture, day)
        _serve(TWTB4U, load_fixture(DAY_TRADES), day, selectType="All")
    DailyChip.objects.create(stock_id="2330", date=DAYS[0], day_trade_volume=1)
    job = {job.key: job for job in JOBS}["twse/day_trade"]
    report = catch_up(DAYS[0], DAYS[-1], october_2026_in_db, [job])
    assert report.written == {"twse/day_trade": [day.isoformat() for day in redone]}
    final = DailyChip.objects.get(stock_id="2330", date=DAYS[0]).day_trade_volume
    assert final == 4_142_000


def test_a_dry_run_lists_the_day_trade_sessions_redone_before_a_missed_night(
    october_2026_in_db: TradingDays,
) -> None:
    for day in DAYS[:2]:
        DailyChip.objects.create(stock_id="2330", date=day, day_trade_volume=1)
    jobs = [job for job in JOBS if job.key in ("twse/day_trade", "twse/margin")]
    planned = plan_catch_up(DAYS[-1], DAYS[-1], october_2026_in_db, jobs)
    assert planned == {
        "twse/day_trade": ["2026-10-27", "2026-10-28", "2026-10-29"],
        "twse/margin": ["2026-10-29"],
    }


def test_sessions_before_the_calendar_are_not_redone(october_2026_in_db: TradingDays) -> None:
    jobs = [job for job in JOBS if job.key == "twse/day_trade"]
    first = date(2026, 10, 1)
    assert plan_catch_up(first, first, october_2026_in_db, jobs) == {
        "twse/day_trade": ["2026-10-01"]
    }


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
    october_2026_in_db: TradingDays,
    load_fixture: Callable[[str], Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses.get(MI_INDEX, json={"stat": "OK", "date": "20261027", "tables": [{"x": 1}]})
    _serve(MI_QFIIS, load_fixture(SHARES), DAYS[0], selectType="ALLBUT0999")
    with caplog.at_level(logging.ERROR, logger="apps.etl.catchup"):
        report = catch_up(DAYS[0], DAYS[0], october_2026_in_db, [PRICE_JOB, SHARE_JOB])
    assert list(report.failed) == ["twse/daily_price 2026-10-27"]
    assert report.written == {"twse/shares_outstanding": ["2026-10-27"]}
    assert [(row.levelname, bool(row.exc_info)) for row in caplog.records] == [("ERROR", False)]


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
    assert list(report.failed) == ["twse/shares_outstanding rebuild after 2026-10-27"]


@responses.activate
def test_the_next_session_checked_against_the_gap_is_checked_again(
    october_2026_in_db: TradingDays,
    load_fixture: Callable[[str], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[set[date]] = []
    monkeypatch.setattr(
        "apps.etl.catchup.dispatch", lambda results: sent.append({row.date for row in results})
    )
    DailyPrice.objects.create(stock_id="2330", date=DAYS[2], volume=1, turnover=1)
    DataQualityLog.objects.create(date=DAYS[2], check_name="row_count", severity="info", detail={})
    _serve_prices(load_fixture, DAYS[1])
    report = catch_up(DAYS[1], DAYS[1], october_2026_in_db, [PRICE_JOB])
    assert report.checked == ["2026-10-28", "2026-10-29"]
    assert sent == [{DAYS[1]}, {DAYS[2]}]


@responses.activate
def test_a_next_session_with_only_other_logs_is_not_checked_again(
    october_2026_in_db: TradingDays,
    load_fixture: Callable[[str], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("apps.etl.catchup.dispatch", lambda results: None)
    DataQualityLog.objects.create(
        date=DAYS[2], check_name="statement_filed_late", severity="info", detail={}
    )
    _serve_prices(load_fixture, DAYS[1])
    report = catch_up(DAYS[1], DAYS[1], october_2026_in_db, [PRICE_JOB])
    assert report.checked == ["2026-10-28"]


@responses.activate
def test_an_unexpected_error_fails_its_job_only(
    october_2026_in_db: TradingDays,
    load_fixture: Callable[[str], Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original = TwseAdapter.normalize

    def normalize(self: TwseAdapter, dataset: str, raw: Any) -> Any:
        if dataset == "daily_price":
            raise ValueError("duplicate keys")
        return original(self, dataset, raw)

    monkeypatch.setattr(TwseAdapter, "normalize", normalize)
    _serve_prices(load_fixture, DAYS[0])
    _serve(MI_QFIIS, load_fixture(SHARES), DAYS[0], selectType="ALLBUT0999")
    with caplog.at_level(logging.ERROR, logger="apps.etl.catchup"):
        report = catch_up(DAYS[0], DAYS[0], october_2026_in_db, [PRICE_JOB, SHARE_JOB])
    assert report.failed == {"twse/daily_price 2026-10-27": "ValueError: duplicate keys"}
    assert report.written == {"twse/shares_outstanding": ["2026-10-27"]}
    assert [bool(row.exc_info) for row in caplog.records] == [True]


@responses.activate
def test_catchup_command_prints_the_report(
    capsys: pytest.CaptureFixture[str], load_fixture: Callable[[str], Any]
) -> None:
    _quoted(load_fixture, DAYS[0])
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
@time_machine.travel(EVENING, tick=False)
def test_catchup_command_rejects_bad_options(options: dict[str, str], message: str) -> None:
    with pytest.raises(CommandError, match=message):
        call_command("catchup", **options)


@responses.activate
@time_machine.travel(EVENING, tick=False)
def test_catchup_command_stops_at_the_session_before_today(
    capsys: pytest.CaptureFixture[str], load_fixture: Callable[[str], Any]
) -> None:
    for day in DAYS[:2]:
        _quoted(load_fixture, day)
    call_command("catchup", since="2026-10-27", source="twse", dataset="daily_price")
    assert capsys.readouterr().out.strip() == "nothing to catch up"


def test_a_price_day_written_without_the_exchanges_quotes_is_fetched_again(
    load_fixture: Callable[[str], Any],
) -> None:
    jobs = {job.key: job for job in JOBS}
    DailyPrice.objects.create(stock_id="2330", date=DAYS[0], volume=1, turnover=1)
    _quoted(load_fixture, DAYS[1])
    assert missing_days(jobs["twse/daily_price"], DAYS) == [DAYS[0], DAYS[2]]


@responses.activate
def test_a_dry_run_lists_the_missing_days_and_fetches_nothing(
    capsys: pytest.CaptureFixture[str], load_fixture: Callable[[str], Any]
) -> None:
    _quoted(load_fixture, DAYS[0])
    call_command(
        "catchup",
        since="2026-10-27",
        until="2026-10-29",
        source="twse",
        dataset="daily_price",
        dry_run=True,
    )
    assert capsys.readouterr().out.strip() == "to fetch twse/daily_price: 2026-10-28, 2026-10-29"
    assert len(responses.calls) == 0
    assert not EtlRun.objects.exists()
