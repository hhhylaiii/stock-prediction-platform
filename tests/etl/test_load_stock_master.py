from collections.abc import Callable
from datetime import date
from io import StringIO
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.models import EtlRun, RawFinmind, RawTpex, RawTwse
from apps.market.models import Market, StockMaster

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TWSE_URL = "https://www.twse.com.tw/rwd/zh/company/suspendListing"
TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/company/deListed"


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    monkeypatch.setattr("apps.etl.adapters.finmind.time.sleep", lambda seconds: None)


@pytest.fixture
def sources(load_fixture: Callable[[str], Any]) -> None:
    responses.get(FINMIND_URL, json=load_fixture("finmind/stock_info.json"))
    responses.get(TWSE_URL, json=load_fixture("twse/reference/suspend_listing.json"))
    responses.post(TPEX_URL, json=load_fixture("tpex/delisted.json"))


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("sources")
def test_command_writes_listed_and_delisted_stocks() -> None:
    call_command("load_stock_master")
    assert StockMaster.objects.count() == 20
    assert StockMaster.objects.filter(delist_date__isnull=False).count() == 6
    tsmc = StockMaster.objects.get(stock_id="2330")
    assert (tsmc.name, tsmc.market, tsmc.industry) == ("台積電", Market.TWSE, "半導體業")
    assert StockMaster.objects.get(stock_id="3454").delist_date == date(2026, 3, 27)
    run = EtlRun.objects.get()
    assert (run.source, run.dataset, run.status, run.rows) == (
        "stock_master",
        "stock_master",
        "success",
        20,
    )
    assert run.checkpoint == {"listed": 14, "delisted": 6, "unlisted_without_delisting": 2}
    assert RawFinmind.objects.filter(dataset="stock_info").count() == 1
    assert RawTwse.objects.filter(dataset="delisted").count() == 1
    assert RawTpex.objects.filter(dataset="delisted").count() == 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("sources")
def test_rerun_is_idempotent_and_keeps_list_dates() -> None:
    call_command("load_stock_master")
    StockMaster.objects.filter(stock_id="2330").update(list_date=date(1994, 9, 5))
    call_command("load_stock_master")
    assert StockMaster.objects.count() == 20
    assert StockMaster.objects.get(stock_id="2330").list_date == date(1994, 9, 5)


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("sources")
def test_relisted_code_loses_its_old_delist_date() -> None:
    StockMaster.objects.create(
        stock_id="6423", name="億而得-創", market=Market.TWSE, delist_date=date(2026, 1, 22)
    )
    call_command("load_stock_master")
    row = StockMaster.objects.get(stock_id="6423")
    assert (row.market, row.delist_date) == (Market.TPEX, None)


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("sources")
def test_stocks_missing_from_every_source_are_kept() -> None:
    StockMaster.objects.create(
        stock_id="1234", name="舊資料", market=Market.TWSE, delist_date=date(1999, 1, 4)
    )
    call_command("load_stock_master")
    assert StockMaster.objects.filter(stock_id="1234").exists()


@responses.activate
@pytest.mark.django_db
def test_source_failure_writes_nothing(load_fixture: Callable[[str], Any]) -> None:
    responses.get(FINMIND_URL, json=load_fixture("finmind/stock_info.json"))
    responses.get(TWSE_URL, json=load_fixture("twse/reference/suspend_listing.json"))
    responses.post(TPEX_URL, status=503)
    with pytest.raises(CommandError, match="RetryableError"):
        call_command("load_stock_master")
    assert StockMaster.objects.count() == 0
    assert EtlRun.objects.get().status == "failed"


@responses.activate
@pytest.mark.django_db
def test_schema_change_is_a_command_error(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture("twse/reference/suspend_listing.json")
    responses.get(FINMIND_URL, json=load_fixture("finmind/stock_info.json"))
    responses.get(TWSE_URL, json={**payload, "fields": ["a", "b", "c"]})
    responses.post(TPEX_URL, json=load_fixture("tpex/delisted.json"))
    with pytest.raises(CommandError, match="SchemaChangedError"):
        call_command("load_stock_master")
    assert StockMaster.objects.count() == 0


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("sources")
def test_the_daily_refresh_count_is_not_the_baseline() -> None:
    EtlRun.objects.create(
        source="stock_master",
        dataset="stock_master_daily",
        start_date=date(2026, 10, 31),
        end_date=date(2026, 10, 31),
        status=EtlRun.Status.SUCCESS,
        checkpoint={"listed": 100000},
    )
    call_command("load_stock_master", stdout=StringIO())
    assert StockMaster.objects.exists()


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("sources")
def test_a_much_shorter_listing_than_last_time_is_refused() -> None:
    EtlRun.objects.create(
        source="stock_master",
        dataset="stock_master",
        start_date=date(2026, 10, 31),
        end_date=date(2026, 10, 31),
        status=EtlRun.Status.SUCCESS,
        checkpoint={"listed": 100, "delisted": 6, "unlisted_without_delisting": 0},
    )
    with pytest.raises(CommandError, match="down from 100"):
        call_command("load_stock_master")
    assert StockMaster.objects.count() == 0


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("sources")
def test_codes_that_vanish_without_a_delisting_are_reported() -> None:
    StockMaster.objects.create(stock_id="0061", name="寶滬深", market=Market.TWSE)
    out = StringIO()
    call_command("load_stock_master", stdout=out)
    assert "3 codes no longer listed and on no delisting list: 0061, 2891A, 9105" in out.getvalue()
    assert StockMaster.objects.get(stock_id="0061").delist_date is None
    assert EtlRun.objects.get().checkpoint["unlisted_without_delisting"] == 3
