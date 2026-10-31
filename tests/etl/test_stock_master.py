from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd
import pytest

from apps.etl.errors import SchemaChangedError
from apps.etl.roc import parse_roc_date
from apps.etl.stock_master import build_stock_master, tpex_delistings, twse_delistings
from apps.market.models import Market


@pytest.fixture
def master(load_fixture: Callable[[str], Any]) -> pd.DataFrame:
    info = load_fixture("finmind/stock_info.json")["data"]
    twse = load_fixture("twse/reference/suspend_listing.json")
    tpex = load_fixture("tpex/delisted.json")["tables"][0]
    frame = build_stock_master(info, twse_delistings(twse), tpex_delistings(tpex))
    return frame.set_index("stock_id")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("115/09/01", date(2026, 9, 1)),
        ("090/01/20", date(2001, 1, 20)),
        ("81-10-27", date(1992, 10, 27)),
        (" 99/01/04", date(2010, 1, 4)),
    ],
)
def test_roc_dates(text: str, expected: date) -> None:
    assert parse_roc_date(text) == expected


@pytest.mark.parametrize("text", ["2026/09/01", "115/13/01", "115/02/30", "", "115-09"])
def test_bad_roc_dates_are_schema_changes(text: str) -> None:
    with pytest.raises(SchemaChangedError):
        parse_roc_date(text)


def test_current_listings_cover_both_markets_and_every_security_type(master: pd.DataFrame) -> None:
    current = master[master["delist_date"].isna()]
    assert set(current.index) == {
        "0050",
        "006201",
        "00679B",
        "01001T",
        "020000",
        "1101",
        "1752",
        "2330",
        "5450",
        "6423",
        "6446",
        "6488",
        "8299",
        "8908",
    }
    assert current.loc["2330", "market"] == Market.TWSE
    assert current.loc["6488", "market"] == Market.TPEX


@pytest.mark.parametrize(
    ("stock_id", "industry"),
    [("2330", "半導體業"), ("6446", "生技醫療業"), ("1752", "生技醫療業"), ("0050", "ETF")],
)
def test_aggregate_industries_give_way_to_the_specific_one(
    master: pd.DataFrame, stock_id: str, industry: str
) -> None:
    assert master.loc[stock_id, "industry"] == industry


@pytest.mark.parametrize(
    ("stock_id", "market"),
    [("6423", Market.TPEX), ("8908", Market.TWSE), ("6446", Market.TWSE), ("1752", Market.TWSE)],
)
def test_board_transfers_stay_listed_on_the_new_market(
    master: pd.DataFrame, stock_id: str, market: str
) -> None:
    assert master.loc[stock_id, "market"] == market
    assert master.loc[stock_id, "delist_date"] is None


def test_reused_code_belongs_to_the_current_company(master: pd.DataFrame) -> None:
    assert (master.loc["5450", "name"], master.loc["5450", "industry"]) == ("南良", "其他")


@pytest.mark.parametrize(
    ("stock_id", "market", "delisted", "name", "industry"),
    [
        ("3454", Market.TWSE, date(2026, 3, 27), "晶睿", "光電業"),
        ("2867", Market.TWSE, date(2026, 9, 1), "三商壽", "金融保險"),
        ("2888", Market.TWSE, date(2025, 7, 24), "新光金", "金融保險"),
        ("4801", Market.TPEX, date(2013, 4, 30), "高盛電子科技股份有限公司", ""),
        ("5901", Market.TPEX, date(2006, 12, 26), "中友百貨", ""),
        ("3126", Market.TPEX, date(2016, 4, 6), "信億科技股份有限公司", ""),
    ],
)
def test_delisted_stocks_keep_their_last_listing(
    master: pd.DataFrame,
    stock_id: str,
    market: str,
    delisted: date,
    name: str,
    industry: str,
) -> None:
    row = master.loc[stock_id]
    assert (row["market"], row["delist_date"], row["name"], row["industry"]) == (
        market,
        delisted,
        name,
        industry,
    )


@pytest.mark.parametrize("stock_id", ["TAIEX", "710566", "9105", "2891A"])
def test_indices_warrants_and_stale_rows_are_left_out(master: pd.DataFrame, stock_id: str) -> None:
    assert stock_id not in master.index


def test_total_rows(master: pd.DataFrame) -> None:
    assert len(master) == 20
    assert master["delist_date"].notna().sum() == 6


def test_changed_twse_fields_are_a_schema_change(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture("twse/reference/suspend_listing.json")
    with pytest.raises(SchemaChangedError, match="fields"):
        twse_delistings({**payload, "fields": ["日期", "名稱", "代號"]})


def test_failed_twse_status_is_a_schema_change(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture("twse/reference/suspend_listing.json")
    with pytest.raises(SchemaChangedError, match="status"):
        twse_delistings({**payload, "status": "error"})


def test_changed_tpex_fields_are_a_schema_change(load_fixture: Callable[[str], Any]) -> None:
    table = load_fixture("tpex/delisted.json")["tables"][0]
    with pytest.raises(SchemaChangedError, match="fields"):
        tpex_delistings({**table, "fields": ["代號", "名稱"]})


def test_malformed_codes_are_a_schema_change(load_fixture: Callable[[str], Any]) -> None:
    table = load_fixture("tpex/delisted.json")["tables"][0]
    broken = {**table, "data": [["12 34", "某公司", "115-01-02", "", ""]]}
    with pytest.raises(SchemaChangedError, match="code"):
        tpex_delistings(broken)


def test_info_rows_without_expected_keys_are_a_schema_change() -> None:
    with pytest.raises(SchemaChangedError, match="TaiwanStockInfo"):
        build_stock_master([{"stock_id": "2330"}], twse_delistings_empty(), tpex_delistings_empty())


def twse_delistings_empty() -> pd.DataFrame:
    return twse_delistings(
        {"status": "ok", "fields": ["終止上市日期", "公司名稱", "上市編號"], "data": []}
    )


def tpex_delistings_empty() -> pd.DataFrame:
    fields = ["股票代號", "公司名稱", "終止上櫃日期", "終止上櫃原因", "公司資料網址"]
    return tpex_delistings({"fields": fields, "data": []})


def test_a_code_on_both_markets_at_once_is_a_schema_change() -> None:
    rows = [
        {"industry_category": "半導體業", "stock_id": "2330", "stock_name": "台積電", "type": t}
        | {"date": "2026-11-01"}
        for t in ("twse", "tpex")
    ]
    with pytest.raises(SchemaChangedError, match="both markets"):
        build_stock_master(rows, twse_delistings_empty(), tpex_delistings_empty())


@pytest.mark.parametrize("market", ["twse", "tpex"])
def test_a_snapshot_missing_a_market_is_a_schema_change(
    load_fixture: Callable[[str], Any], market: str
) -> None:
    info = [
        {**row, "type": row["type"].upper()} if row["type"] == market else row
        for row in load_fixture("finmind/stock_info.json")["data"]
    ]
    with pytest.raises(SchemaChangedError, match=f"no current {market}"):
        build_stock_master(info, twse_delistings_empty(), tpex_delistings_empty())
