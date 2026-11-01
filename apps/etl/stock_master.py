"""stock_master rows from current listings plus the official delisting lists (SPEC §5, §8.3)."""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from apps.etl.adapters.twse import is_foreign_currency_etf
from apps.etl.errors import SchemaChangedError
from apps.etl.exchange_rows import stock_code
from apps.etl.roc import parse_roc_date
from apps.market.models import Market

TWSE_DELISTING_FIELDS = ["終止上市日期", "公司名稱", "上市編號"]
TPEX_DELISTING_FIELDS = ["股票代號", "公司名稱", "終止上櫃日期", "終止上櫃原因", "公司資料網址"]
INFO_KEYS = frozenset({"industry_category", "stock_id", "stock_name", "type", "date"})
INFO_MARKETS = {"twse": Market.TWSE, "tpex": Market.TPEX}
EXCLUDED_INDUSTRIES = frozenset({"Index", "大盤", "所有證券"})
AGGREGATE_INDUSTRIES = frozenset({"電子工業", "化學生技醫療", "創新板股票", "創新版股票"})
DELISTING_COLUMNS = ["stock_id", "name", "market", "delist_date"]
MASTER_COLUMNS = ["stock_id", "name", "market", "industry", "delist_date"]
COMPANY_SUFFIX = "股份有限公司"
LISTED_DROP_LIMIT = 0.05


def check_listed_count(listed: int, previous: int | None) -> None:
    """A snapshot listing far fewer stocks than the last load is taken as incomplete."""
    if previous and listed < previous * (1 - LISTED_DROP_LIMIT):
        raise SchemaChangedError(f"TaiwanStockInfo lists {listed} stocks, down from {previous}")


def pick_industry(categories: Sequence[str]) -> str:
    """First specific category in the given order; aggregates like 電子工業 only as a fallback."""
    specific = [name for name in categories if name not in AGGREGATE_INDUSTRIES]
    candidates = specific or list(categories)
    return candidates[0] if candidates else ""


def _delistings(
    rows: Any, label: str, market: str, code_at: int, name_at: int, date_at: int
) -> pd.DataFrame:
    if not isinstance(rows, list) or not rows:
        raise SchemaChangedError(f"{label} delisting list has no rows")
    records = [
        {
            "stock_id": stock_code(row[code_at]),
            "name": str(row[name_at]).strip().removesuffix(COMPANY_SUFFIX),
            "market": market,
            "delist_date": parse_roc_date(row[date_at]),
        }
        for row in rows
    ]
    return pd.DataFrame(records, columns=DELISTING_COLUMNS)


def twse_delistings(payload: dict[str, Any]) -> pd.DataFrame:
    if payload.get("status") != "ok":
        raise SchemaChangedError(f"TWSE delisting list status is {payload.get('status')!r}")
    if payload.get("fields") != TWSE_DELISTING_FIELDS:
        raise SchemaChangedError(f"TWSE delisting list fields changed: {payload.get('fields')}")
    return _delistings(payload.get("data"), "TWSE", Market.TWSE, code_at=2, name_at=1, date_at=0)


def tpex_delistings(table: dict[str, Any]) -> pd.DataFrame:
    if table.get("fields") != TPEX_DELISTING_FIELDS:
        raise SchemaChangedError(f"TPEx delisting list fields changed: {table.get('fields')}")
    return _delistings(table.get("data"), "TPEx", Market.TPEX, code_at=0, name_at=1, date_at=2)


def _listed_info(info_rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    info = pd.DataFrame(list(info_rows))
    missing = INFO_KEYS - set(info.columns)
    if missing:
        raise SchemaChangedError(f"TaiwanStockInfo rows lost keys {sorted(missing)}")
    keep = (
        info["type"].isin(list(INFO_MARKETS))
        & ~info["industry_category"].isin(list(EXCLUDED_INDUSTRIES))
        & (info["date"] != "None")
        & ~info["stock_id"].astype(str).str.strip().map(is_foreign_currency_etf)
    )
    return info.loc[keep].sort_values(["stock_id", "date"], ascending=[True, False])


def _current(info: pd.DataFrame) -> pd.DataFrame:
    latest = info.loc[info["date"] == info["date"].max()]
    records = []
    for code, rows in latest.groupby("stock_id", sort=True):
        markets = set(rows["type"])
        if len(markets) != 1:
            raise SchemaChangedError(f"{code} is listed on both markets: {sorted(markets)}")
        records.append(
            {
                "stock_id": stock_code(code),
                "name": str(rows["stock_name"].iloc[0]),
                "market": INFO_MARKETS[markets.pop()],
                "industry": pick_industry(sorted(rows["industry_category"])),
                "delist_date": None,
            }
        )
    current = pd.DataFrame(records, columns=MASTER_COLUMNS)
    for source, market in INFO_MARKETS.items():
        if not (current["market"] == market).any():
            raise SchemaChangedError(f"TaiwanStockInfo has no current {source} listings")
    return current


def _delisted(info: pd.DataFrame, delistings: pd.DataFrame, current: set[str]) -> pd.DataFrame:
    latest: dict[str, dict[str, Any]] = {}
    for record in delistings.sort_values("delist_date").to_dict("records"):
        latest[str(record["stock_id"])] = {str(key): value for key, value in record.items()}
    history: dict[str, list[dict[str, Any]]] = {}
    for record in info.to_dict("records"):
        row = {str(key): value for key, value in record.items()}
        history.setdefault(str(row["stock_id"]), []).append(row)
    records = []
    for code in sorted(set(latest) - current):
        known = history.get(code, [])
        records.append(
            {
                "stock_id": code,
                "name": str(known[0]["stock_name"]) if known else latest[code]["name"],
                "market": latest[code]["market"],
                "industry": pick_industry([str(row["industry_category"]) for row in known]),
                "delist_date": latest[code]["delist_date"],
            }
        )
    return pd.DataFrame(records, columns=MASTER_COLUMNS)


def unlisted_codes(info_rows: Sequence[dict[str, Any]], master: pd.DataFrame) -> list[str]:
    """Codes FinMind once listed that are neither listed today nor on a delisting list."""
    history = set(_listed_info(info_rows)["stock_id"].astype(str).str.strip())
    return sorted(history - set(master["stock_id"]))


def build_stock_master(
    info_rows: Sequence[dict[str, Any]], twse: pd.DataFrame, tpex: pd.DataFrame
) -> pd.DataFrame:
    """Latest listings plus officially delisted codes; a code listed today is never delisted."""
    info = _listed_info(info_rows)
    current = _current(info)
    delisted = _delisted(info, pd.concat([twse, tpex], ignore_index=True), set(current["stock_id"]))
    master = pd.concat([current, delisted], ignore_index=True)
    return master.sort_values("stock_id", ignore_index=True)
