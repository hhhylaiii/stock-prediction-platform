from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.models import DataQualityLog, EtlRun, RawTpex, RawTwse
from apps.etl.raw import store_raw
from apps.market.models import DailyPrice, Market, StockMaster

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
@pytest.mark.usefixtures("quotes", "delistings")
def test_a_code_newly_missing_for_five_sessions_is_alerted(
    alerts: list[list[DataQualityLog]],
) -> None:
    _master("9999", Market.TWSE)
    _master("8888", Market.TWSE)
    _master("7777", Market.TWSE)
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
