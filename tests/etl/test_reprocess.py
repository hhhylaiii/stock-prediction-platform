from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.adapters.twse import TwseAdapter
from apps.etl.errors import SourceNotReadyError
from apps.etl.models import EtlRun, RawTwse
from apps.etl.ratelimit import RateLimiter
from apps.etl.replay import reprocess
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, DispositionEvent, SharesOutstanding, StockMaster

TWSE = "https://www.twse.com.tw/rwd/zh"
TPEX = "https://www.tpex.org.tw/www/zh-tw"
DAY = date(2026, 10, 28)
NEXT = date(2026, 10, 29)
LATER = date(2026, 10, 30)
NOTICE = f"{TWSE}/announcement/notice"
TWT85U = "https://www.twse.com.tw/exchangeReport/TWT85U"


@pytest.fixture
def twse(october_2026: TradingDays, twse_limiter: RateLimiter) -> TwseAdapter:
    return TwseAdapter(october_2026, twse_limiter)


@pytest.fixture
def tpex(october_2026: TradingDays, twse_limiter: RateLimiter) -> TpexAdapter:
    return TpexAdapter(october_2026, twse_limiter)


def _load_twse_prices(twse: TwseAdapter, load_fixture: Callable[[str], Any]) -> None:
    responses.get(
        f"{TWSE}/afterTrading/MI_INDEX",
        json=load_fixture("twse/daily_price/mi_index_20261028.json"),
    )
    twse.run("daily_price", DAY, DAY)


def _load_twse_shares(
    twse: TwseAdapter, load_fixture: Callable[[str], Any], name: str, day: date
) -> None:
    responses.get(f"{TWSE}/fund/MI_QFIIS", json=load_fixture(f"twse/shares_outstanding/{name}"))
    twse.run("shares_outstanding", day, day)


@responses.activate
@pytest.mark.django_db
def test_prices_are_rebuilt_from_raw_without_any_request(
    twse: TwseAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    _load_twse_prices(twse, load_fixture)
    stored = {row.stock_id: row.close for row in DailyPrice.objects.all()}
    DailyPrice.objects.filter(stock_id="2330").update(close=Decimal("1"))
    DailyPrice.objects.filter(stock_id="1101").delete()
    responses.reset()
    run = reprocess("twse", "daily_price", DAY, DAY, october_2026).run
    assert {row.stock_id: row.close for row in DailyPrice.objects.all()} == stored
    assert len(responses.calls) == 0
    assert (run.status, run.checkpoint["kind"], run.checkpoint["last_date"]) == (
        "success",
        "reprocess",
        "2026-10-28",
    )


@responses.activate
@pytest.mark.django_db
def test_a_day_without_stored_raw_stops_the_run_after_the_days_before_it(
    twse: TwseAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    _load_twse_prices(twse, load_fixture)
    DailyPrice.objects.all().delete()
    responses.reset()
    with pytest.raises(SourceNotReadyError, match="no stored raw"):
        reprocess("twse", "daily_price", DAY, date(2026, 10, 29), october_2026)
    run = EtlRun.objects.filter(checkpoint__kind="reprocess").get()
    assert (run.status, run.checkpoint["last_date"]) == ("failed", "2026-10-28")
    assert DailyPrice.objects.filter(date=DAY).exists()


@responses.activate
@pytest.mark.django_db
def test_posted_requests_replay_too(
    tpex: TpexAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    responses.post(
        f"{TPEX}/afterTrading/otc", json=load_fixture("tpex/daily_price/otc_20261028.json")
    )
    responses.post(
        f"{TPEX}/afterTrading/dailyQuotes",
        json=load_fixture("tpex/daily_price/daily_quotes_20261028.json"),
    )
    tpex.run("daily_price", DAY, DAY)
    stored = DailyPrice.objects.count()
    DailyPrice.objects.all().delete()
    responses.reset()
    reprocess("tpex", "daily_price", DAY, DAY, october_2026)
    assert DailyPrice.objects.count() == stored
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.django_db
def test_share_rows_of_the_codes_listed_that_day_are_cleared_before_they_are_rebuilt(
    twse: TwseAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code, name=code, market="TWSE") for code in ("0050", "2330", "2368")
    )
    StockMaster.objects.create(stock_id="6488", name="6488", market="TWSE")
    _load_twse_shares(twse, load_fixture, "mi_qfiis_20261028.json", DAY)
    _load_twse_shares(twse, load_fixture, "mi_qfiis_20261030.json", LATER)
    rebuilt = set(SharesOutstanding.objects.values_list("stock_id", "effective_date", "shares"))
    SharesOutstanding.objects.create(stock_id="2330", effective_date=LATER, shares=1)
    SharesOutstanding.objects.create(stock_id="6488", effective_date=LATER, shares=7)
    responses.reset()
    reprocess("twse", "shares_outstanding", LATER, LATER, october_2026)
    after = set(SharesOutstanding.objects.values_list("stock_id", "effective_date", "shares"))
    assert after == rebuilt | {("6488", LATER, 7)}


@responses.activate
@pytest.mark.django_db
def test_a_monthly_dataset_fetched_day_by_day_replays_day_by_day(
    twse: TwseAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    for day in (DAY, NEXT):
        responses.get(NOTICE, json=load_fixture("twse/disposition/notice_20261028.json"))
        twse.run("attention", day, day)
    events = set(DispositionEvent.objects.values_list("stock_id", "start_date", "type"))
    DispositionEvent.objects.all().delete()
    responses.reset()
    run = reprocess("twse", "attention", DAY, NEXT, october_2026).run
    assert set(DispositionEvent.objects.values_list("stock_id", "start_date", "type")) == events
    assert (run.status, run.checkpoint["last_date"]) == ("success", "2026-10-29")
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.django_db
def test_day_events_of_the_market_are_cleared_before_they_are_rebuilt(
    twse: TwseAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code, name=code, market="TWSE") for code in ("1213", "9999")
    )
    StockMaster.objects.create(stock_id="6488", name="6488", market="TPEx")
    responses.get(TWT85U, json=load_fixture("twse/disposition/twt85u_20261028.json"))
    twse.run("full_delivery", DAY, DAY)
    rebuilt = set(DispositionEvent.objects.values_list("stock_id", "type"))
    for code, kind in (
        ("9999", "full_delivery"),
        ("6488", "full_delivery"),
        ("9999", "disposition"),
    ):
        DispositionEvent.objects.create(stock_id=code, start_date=DAY, type=kind, end_date=DAY)
    responses.reset()
    reprocess("twse", "full_delivery", DAY, DAY, october_2026)
    left = set(DispositionEvent.objects.values_list("stock_id", "type"))
    assert left == rebuilt | {("6488", "full_delivery"), ("9999", "disposition")}


def _with_count(payload: dict[str, Any], code: str, count: str) -> dict[str, Any]:
    rows = [[*row[:3], count, *row[4:]] if row[0] == code else row for row in payload["data"]]
    return {**payload, "data": rows}


@responses.activate
@pytest.mark.django_db
def test_the_latest_stored_answer_is_replayed(
    twse: TwseAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture("twse/daily_price/mi_index_20261028.json")
    _load_twse_prices(twse, load_fixture)
    first = DailyPrice.objects.get(stock_id="2330").close
    table = next(t for t in payload["tables"] if "每日收盤行情" in (t.get("title") or ""))
    close_at = table["fields"].index("收盤價")
    for row in table["data"]:
        if row[0] == "2330":
            row[close_at] = "1,500.00"
    responses.get(f"{TWSE}/afterTrading/MI_INDEX", json=payload)
    twse.run("daily_price", DAY, DAY)
    DailyPrice.objects.filter(stock_id="2330").update(close=first)
    responses.reset()
    reprocess("twse", "daily_price", DAY, DAY, october_2026)
    assert DailyPrice.objects.get(stock_id="2330").close == Decimal("1500")


@responses.activate
@pytest.mark.django_db
def test_a_corrected_share_count_rebuilds_the_later_fetched_days(
    twse: TwseAdapter, october_2026: TradingDays, load_fixture: Callable[[str], Any]
) -> None:
    StockMaster.objects.create(stock_id="2330", name="2330", market="TWSE")
    payload = load_fixture("twse/shares_outstanding/mi_qfiis_20261028.json")
    early = _with_count(payload, "2330", "99,000")
    for day, answer in ((date(2026, 10, 27), payload), (DAY, early), (NEXT, early)):
        responses.get(f"{TWSE}/fund/MI_QFIIS", json={**answer, "date": f"{day:%Y%m%d}"})
        twse.run("shares_outstanding", day, day)
        responses.reset()
    responses.get(f"{TWSE}/fund/MI_QFIIS", json=payload)
    twse.run("shares_outstanding", DAY, DAY)
    responses.reset()
    later = reprocess("twse", "shares_outstanding", DAY, DAY, october_2026).rebuilt
    counts = dict(
        SharesOutstanding.objects.filter(stock_id="2330").values_list("effective_date", "shares")
    )
    assert later == [NEXT]
    assert counts == {date(2026, 10, 27): 25932370067, NEXT: 99000}


@responses.activate
@pytest.mark.django_db
def test_a_day_twse_listed_no_notices_replays_as_an_empty_list(
    twse: TwseAdapter, october_2026: TradingDays
) -> None:
    responses.get(NOTICE, json={"stat": "很抱歉，沒有符合條件的資料!"})
    twse.run("attention", DAY, DAY)
    responses.reset()
    run = reprocess("twse", "attention", DAY, DAY, october_2026).run
    assert (run.status, run.rows) == ("success", 0)


@responses.activate
@pytest.mark.django_db
def test_attention_rows_are_kept_since_the_next_days_answer_may_add_to_them(
    twse: TwseAdapter, october_2026: TradingDays
) -> None:
    StockMaster.objects.create(stock_id="2330", name="2330", market="TWSE")
    responses.get(NOTICE, json={"stat": "很抱歉，沒有符合條件的資料!"})
    twse.run("attention", DAY, DAY)
    DispositionEvent.objects.create(stock_id="2330", start_date=DAY, type="attention", end_date=DAY)
    responses.reset()
    reprocess("twse", "attention", DAY, DAY, october_2026)
    assert DispositionEvent.objects.filter(stock_id="2330").exists()


@pytest.mark.django_db
@pytest.mark.parametrize("source", ["finmind", "yahoo", "tdcc", "datagov"])
def test_sources_without_dated_requests_cannot_be_reprocessed(
    october_2026: TradingDays, source: str
) -> None:
    with pytest.raises(ValueError, match="reprocess"):
        reprocess(source, "daily_price", DAY, DAY, october_2026)


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_reprocess_command_rebuilds_the_range(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    responses.get(
        f"{TWSE}/afterTrading/MI_INDEX",
        json=load_fixture("twse/daily_price/mi_index_20261028.json"),
    )
    call_command("fetch", source="twse", dataset="daily_price", date="2026-10-28")
    DailyPrice.objects.all().delete()
    responses.reset()
    call_command(
        "reprocess", source="twse", dataset="daily_price", start="2026-10-28", end="2026-10-28"
    )
    assert DailyPrice.objects.filter(date=DAY).exists()
    assert RawTwse.objects.count() == 1


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_reprocess_command_lists_the_later_share_days_it_rebuilt(
    load_fixture: Callable[[str], Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    payload = load_fixture("twse/shares_outstanding/mi_qfiis_20261028.json")
    for day in ("20261028", "20261029"):
        responses.get(f"{TWSE}/fund/MI_QFIIS", json={**payload, "date": day})
        call_command(
            "fetch",
            source="twse",
            dataset="shares_outstanding",
            date=day[:4] + "-" + day[4:6] + "-" + day[6:],
        )
        responses.reset()
    call_command(
        "reprocess",
        source="twse",
        dataset="shares_outstanding",
        start="2026-10-28",
        end="2026-10-28",
    )
    assert capsys.readouterr().out.splitlines()[-1] == "later fetched days rebuilt: 2026-10-29"


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_a_failed_later_rebuild_is_reported_after_the_range(
    load_fixture: Callable[[str], Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    payload = load_fixture("twse/shares_outstanding/mi_qfiis_20261028.json")
    for day in (DAY, NEXT):
        responses.get(f"{TWSE}/fund/MI_QFIIS", json={**payload, "date": f"{day:%Y%m%d}"})
        call_command("fetch", source="twse", dataset="shares_outstanding", date=day.isoformat())
        responses.reset()
    RawTwse.objects.filter(request_params__date="20261029").delete()
    with pytest.raises(CommandError, match="fetched days after 2026-10-28 failed: MissingRawError"):
        call_command(
            "reprocess",
            source="twse",
            dataset="shares_outstanding",
            start="2026-10-28",
            end="2026-10-28",
        )
    assert "success" in capsys.readouterr().out
