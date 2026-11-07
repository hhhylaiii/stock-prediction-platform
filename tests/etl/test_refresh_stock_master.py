from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.models import DataQualityLog, EtlRun, RawTpex, RawTwse
from apps.etl.raw import store_raw
from apps.market.models import DailyPrice, Market, MarketHistory, StockMaster

TWSE_URL = "https://www.twse.com.tw/rwd/zh/company/suspendListing"
TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/company/deListed"
DAY = date(2026, 10, 28)

pytestmark = pytest.mark.usefixtures("october_2026_in_db", "quiet")


@pytest.fixture
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)


@pytest.fixture
def alerts(monkeypatch: pytest.MonkeyPatch) -> list[list[DataQualityLog]]:
    sent: list[list[DataQualityLog]] = []
    monkeypatch.setattr("apps.etl.master_refresh.dispatch", lambda rows: sent.append(list(rows)))
    return sent


@pytest.fixture
def quotes(db: None, load_fixture: Callable[[str], Any]) -> None:
    store_raw(
        RawTwse,
        "daily_price",
        {"date": "20261028", "type": "ALLBUT0999", "response": "json"},
        load_fixture("twse/daily_price/mi_index_20261028.json"),
    )
    store_raw(
        RawTpex,
        "daily_price",
        {"date": "2026/10/28", "type": "EW", "response": "json"},
        load_fixture("tpex/daily_price/otc_20261028.json"),
    )


@pytest.fixture
def delistings(load_fixture: Callable[[str], Any]) -> None:
    responses.get(TWSE_URL, json=load_fixture("twse/reference/suspend_listing.json"))
    responses.post(TPEX_URL, json=load_fixture("tpex/delisted.json"))


def _master(code: str, market: str, name: str = "", delisted: date | None = None) -> None:
    StockMaster.objects.create(
        stock_id=code, name=name or code, market=market, industry="半導體業", delist_date=delisted
    )


def _state() -> dict[str, tuple[str, str, date | None]]:
    rows = StockMaster.objects.values_list("stock_id", "name", "market", "delist_date")
    return {code: (name, market, delisted) for code, name, market, delisted in rows}


@responses.activate
@pytest.mark.usefixtures("quotes", "delistings", "alerts")
def test_the_days_quotes_and_recent_delistings_update_stock_master() -> None:
    _master("2330", Market.TWSE, "台積電")
    _master("1101", Market.TWSE, "舊名")
    _master("3293", Market.TWSE, "鈊象")
    _master("2317", Market.TWSE, "鴻海", delisted=date(2020, 1, 2))
    _master("8908", Market.TPEX, "欣雄")
    _master("6446", Market.TPEX, "藥華藥")
    call_command("refresh_stock_master", date="2026-10-28")
    state = _state()
    assert state["2330"] == ("台積電", Market.TWSE, None)
    assert state["1101"] == ("台泥", Market.TWSE, None)
    assert state["3293"] == ("鈊象", Market.TPEX, None)
    assert state["2317"] == ("鴻海", Market.TWSE, None)
    assert state["8908"] == ("欣雄", Market.TPEX, date(2026, 10, 16))
    assert state["6446"] == ("藥華藥", Market.TPEX, None)
    assert state["1240"] == ("茂生農經", Market.TPEX, None)
    assert state["0050"] == ("元大台灣50", Market.TWSE, None)
    assert "00636K" not in state
    assert "00687C" not in state
    assert StockMaster.objects.get(stock_id="3293").industry == "半導體業"
    assert StockMaster.objects.get(stock_id="1240").industry == ""
    checkpoint = EtlRun.objects.get(dataset="stock_master_daily").checkpoint
    assert checkpoint["moved"] == ["3293"]
    assert checkpoint["relisted"] == ["2317"]
    assert checkpoint["delisted"] == ["8908"]
    assert "1240" in checkpoint["added"]
    assert checkpoint["listed"] == StockMaster.objects.filter(delist_date__isnull=True).count()


@responses.activate
@pytest.mark.usefixtures("quotes", "delistings", "alerts")
def test_the_days_quotes_open_market_spans_for_moved_and_new_codes() -> None:
    _master("3293", Market.TWSE, "鈊象")
    _master("2330", Market.TWSE, "台積電")
    for code in ("3293", "2330"):
        MarketHistory.objects.create(stock_id=code, start_date=date(2010, 1, 4), market=Market.TWSE)
    call_command("refresh_stock_master", date="2026-10-28")
    spans = MarketHistory.objects.order_by("stock_id", "start_date")
    rows = list(spans.values_list("stock_id", "start_date", "end_date", "market"))
    assert ("3293", date(2010, 1, 4), date(2026, 10, 27), Market.TWSE) in rows
    assert ("3293", DAY, None, Market.TPEX) in rows
    assert ("2330", date(2010, 1, 4), None, Market.TWSE) in rows
    opened = set(spans.filter(end_date__isnull=True).values_list("stock_id", flat=True))
    assert opened == set(StockMaster.objects.values_list("stock_id", flat=True))


@responses.activate
@pytest.mark.usefixtures("quotes", "delistings")
def test_a_code_newly_missing_for_five_sessions_is_alerted(
    alerts: list[list[DataQualityLog]],
) -> None:
    _master("9999", Market.TWSE)
    _master("8888", Market.TWSE)
    _master("7777", Market.TWSE)
    _master("6666", Market.TWSE)
    DailyPrice.objects.create(stock_id="6666", date=date(2026, 10, 16), volume=1, turnover=1)
    DailyPrice.objects.create(stock_id="9999", date=date(2026, 10, 19), volume=1, turnover=1)
    for day in (date(2026, 10, 19), date(2026, 10, 27)):
        DailyPrice.objects.create(stock_id="7777", date=day, volume=1, turnover=1)
    call_command("refresh_stock_master", date="2026-10-28")
    row = DataQualityLog.objects.get(date=DAY, check_name="stock_master")
    assert (row.severity, row.detail["vanished"]) == ("warning", ["9999"])
    assert [[sent.check_name for sent in batch] for batch in alerts] == [["stock_master"]]
    assert _state()["9999"][2] is None


@responses.activate
@pytest.mark.usefixtures("quotes", "delistings", "alerts")
def test_a_rerun_changes_nothing_more_and_keeps_one_log_row() -> None:
    _master("3293", Market.TWSE)
    _master("8908", Market.TPEX)
    call_command("refresh_stock_master", date="2026-10-28")
    first = _state()
    call_command("refresh_stock_master", date="2026-10-28")
    assert _state() == first
    last = EtlRun.objects.filter(dataset="stock_master_daily").order_by("-run_id").first()
    assert last is not None
    found = last.checkpoint
    assert (found["added"], found["moved"], found["delisted"]) == ([], [], [])
    row = DataQualityLog.objects.get(date=DAY, check_name="stock_master")
    assert row.severity == "info"


@responses.activate
@pytest.mark.usefixtures("delistings", "alerts")
def test_quotes_not_loaded_yet_change_nothing(load_fixture: Callable[[str], Any]) -> None:
    store_raw(
        RawTwse,
        "daily_price",
        {"date": "20261028", "type": "ALLBUT0999", "response": "json"},
        load_fixture("twse/daily_price/mi_index_20261028.json"),
    )
    _master("3293", Market.TWSE)
    with pytest.raises(CommandError, match="TPEx daily_price for 2026-10-28 is not loaded"):
        call_command("refresh_stock_master", date="2026-10-28")
    assert _state() == {"3293": ("3293", Market.TWSE, None)}
    assert EtlRun.objects.get().status == "failed"
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.usefixtures("quotes", "delistings", "alerts")
def test_a_code_quoted_on_both_exchanges_is_refused(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture("tpex/daily_price/otc_20261028.json")
    for table in payload["tables"]:
        if "收盤行情" in (table.get("title") or ""):
            table["data"] = [["2330", "台積電", *table["data"][0][2:]], *table["data"]]
            table["totalCount"] = len(table["data"])
    store_raw(RawTpex, "daily_price", {"date": "2026/10/28", "type": "EW", "x": "1"}, payload)
    with pytest.raises(CommandError, match="2330 is quoted on both"):
        call_command("refresh_stock_master", date="2026-10-28")
    assert not StockMaster.objects.exists()


def test_a_day_that_is_not_a_trading_day_is_refused() -> None:
    with pytest.raises(CommandError, match="not a trading day"):
        call_command("refresh_stock_master", date="2026-10-25")


@responses.activate
@pytest.mark.usefixtures("quotes", "delistings", "alerts")
def test_a_day_before_the_last_refresh_is_refused() -> None:
    call_command("refresh_stock_master", date="2026-10-28")
    with pytest.raises(CommandError, match="before the last refresh on 2026-10-28"):
        call_command("refresh_stock_master", date="2026-10-27")


@responses.activate
@pytest.mark.usefixtures("quotes", "delistings", "alerts")
def test_a_delisting_after_the_day_is_kept_for_a_quoted_code() -> None:
    _master("1101", Market.TWSE, "台泥", delisted=date(2026, 10, 30))
    call_command("refresh_stock_master", date="2026-10-28")
    assert _state()["1101"] == ("台泥", Market.TWSE, date(2026, 10, 30))
    assert EtlRun.objects.get().checkpoint["relisted"] == []


@responses.activate
@pytest.mark.usefixtures("quotes", "alerts")
def test_only_a_recent_delisting_by_the_codes_own_market_applies(
    load_fixture: Callable[[str], Any],
) -> None:
    responses.get(TWSE_URL, json=load_fixture("twse/reference/suspend_listing.json"))
    tpex = load_fixture("tpex/delisted.json")
    table = tpex["tables"][0]
    later = ["6666", "後下市股份有限公司", "115-10-30", *table["data"][0][3:]]
    table["data"] = [*table["data"], later]
    table["totalCount"] = len(table["data"])
    responses.post(TPEX_URL, json=tpex)
    _master("8908", Market.TWSE, "欣雄")
    _master("6666", Market.TPEX, "後下市")
    call_command("refresh_stock_master", date="2026-10-28")
    assert _state()["8908"][2] is None
    assert _state()["6666"][2] is None


def _quotes_for(load_fixture: Callable[[str], Any], day: date) -> None:
    twse = {**load_fixture("twse/daily_price/mi_index_20261028.json"), "date": f"{day:%Y%m%d}"}
    tpex = {**load_fixture("tpex/daily_price/otc_20261028.json"), "date": f"{day:%Y%m%d}"}
    store_raw(RawTwse, "daily_price", {"date": f"{day:%Y%m%d}", "type": "ALLBUT0999"}, twse)
    store_raw(RawTpex, "daily_price", {"date": f"{day:%Y/%m/%d}", "type": "EW"}, tpex)


@responses.activate
@pytest.mark.usefixtures("delistings")
def test_codes_gone_quiet_on_a_skipped_day_are_still_reported_once(
    load_fixture: Callable[[str], Any], alerts: list[list[DataQualityLog]]
) -> None:
    EtlRun.objects.create(
        source="stock_master",
        dataset="stock_master_daily",
        start_date=date(2026, 10, 27),
        end_date=date(2026, 10, 27),
        status=EtlRun.Status.SUCCESS,
    )
    _quotes_for(load_fixture, date(2026, 10, 29))
    for code, last in (("5551", date(2026, 10, 16)), ("5552", date(2026, 10, 19))):
        _master(code, Market.TWSE)
        DailyPrice.objects.create(stock_id=code, date=last, volume=1, turnover=1)
    _master("5553", Market.TWSE)
    DailyPrice.objects.create(stock_id="5553", date=date(2026, 10, 20), volume=1, turnover=1)
    call_command("refresh_stock_master", date="2026-10-29")
    row = DataQualityLog.objects.get(date=date(2026, 10, 29), check_name="stock_master")
    assert row.detail["vanished"] == ["5552", "5553"]


@responses.activate
@pytest.mark.usefixtures("delistings", "alerts")
def test_a_renamed_twse_name_column_is_a_schema_change(
    load_fixture: Callable[[str], Any],
) -> None:
    payload = load_fixture("twse/daily_price/mi_index_20261028.json")
    for table in payload["tables"]:
        if "每日收盤行情" in (table.get("title") or ""):
            table["fields"] = [
                "證券簡稱" if name == "證券名稱" else name for name in table["fields"]
            ]
    store_raw(RawTwse, "daily_price", {"date": "20261028", "type": "ALLBUT0999"}, payload)
    store_raw(
        RawTpex,
        "daily_price",
        {"date": "2026/10/28", "type": "EW"},
        load_fixture("tpex/daily_price/otc_20261028.json"),
    )
    with pytest.raises(CommandError, match="TWSE daily quotes fields changed"):
        call_command("refresh_stock_master", date="2026-10-28")


@responses.activate
@pytest.mark.usefixtures("quotes", "alerts")
def test_a_later_delisting_by_the_other_market_does_not_hide_its_own(
    load_fixture: Callable[[str], Any],
) -> None:
    twse = load_fixture("twse/reference/suspend_listing.json")
    twse["data"] = [*twse["data"], ["115/10/25", "轉上市", "7777"]]
    responses.get(TWSE_URL, json=twse)
    tpex = load_fixture("tpex/delisted.json")
    table = tpex["tables"][0]
    table["data"] = [
        *table["data"],
        ["7777", "轉上市股份有限公司", "115-10-20", *table["data"][0][3:]],
    ]
    table["totalCount"] = len(table["data"])
    responses.post(TPEX_URL, json=tpex)
    _master("7777", Market.TPEX)
    call_command("refresh_stock_master", date="2026-10-28")
    assert _state()["7777"][2] == date(2026, 10, 20)
