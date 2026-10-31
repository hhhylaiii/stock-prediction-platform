"""FinMind rows mapped onto the standard tables; units checked against TWSE (SPEC §4)."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd

from apps.etl.errors import SchemaChangedError
from apps.market.models import ShareChangeReason

logger = logging.getLogger(__name__)
Row = Mapping[str, Any]
PRICE_KEYS = ("date", "stock_id", "open", "max", "min", "close", "Trading_Volume", "Trading_money")
PRICE_COLUMNS = ["stock_id", "date", "open", "high", "low", "close", "volume", "turnover"]
POLICY_KEYS = (
    "stock_id",
    "StockEarningsDistribution",
    "StockStatutorySurplus",
    "StockExDividendTradingDate",
    "CashEarningsDistribution",
    "CashStatutorySurplus",
    "CashExDividendTradingDate",
)
RESULT_KEYS = ("date", "stock_id", "before_price", "after_price")
EVENT_COLUMNS = [
    "ex_date",
    "stock_id",
    "cash_dividend",
    "stock_dividend",
    "adj_factor",
]
VALUATION_KEYS = ("date", "stock_id", "PER", "PBR", "dividend_yield")
VALUATION_COLUMNS = ["stock_id", "date", "pe", "pb", "dividend_yield"]
SHARE_KEYS = ("date", "stock_id", "NumberOfSharesIssued")
SHARE_COLUMNS = ["stock_id", "effective_date", "shares", "reason"]
DIVIDEND_PARTS = (
    ("stock", "StockExDividendTradingDate", ("StockEarningsDistribution", "StockStatutorySurplus")),
    ("cash", "CashExDividendTradingDate", ("CashEarningsDistribution", "CashStatutorySurplus")),
)
DIVIDEND_PLACES = Decimal("0.00000001")
PERCENT = 100.0


def _checked(rows: Iterable[Row], keys: Sequence[str], label: str, stock_id: str) -> list[Row]:
    checked = []
    for row in rows:
        missing = [key for key in keys if key not in row]
        if missing:
            raise SchemaChangedError(f"FinMind {label} rows lost keys {missing}")
        if str(row["stock_id"]).strip() != stock_id:
            raise SchemaChangedError(f"FinMind {label} returned {row['stock_id']!r} for {stock_id}")
        checked.append(row)
    return checked


def _day(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        raise SchemaChangedError(f"FinMind {label} has a bad date {value!r}") from None


def _optional_day(value: Any, label: str) -> date | None:
    return _day(value, label) if str(value or "").strip() else None


def _decimal(value: Any, label: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise SchemaChangedError(f"FinMind {label} has a bad number {value!r}") from None
    if not number.is_finite() or number < 0:
        raise SchemaChangedError(f"FinMind {label} has a bad number {value!r}")
    return number


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise SchemaChangedError(f"FinMind {label} has a bad count {value!r}")
    return int(value)


def price_bars(rows: Iterable[Row], stock_id: str) -> pd.DataFrame:
    """TaiwanStockPrice → daily_price; OHLC of 0 marks a day without trades."""
    records = []
    for row in _checked(rows, PRICE_KEYS, "TaiwanStockPrice", stock_id):
        prices = [_decimal(row[key], "TaiwanStockPrice") for key in ("open", "max", "min", "close")]
        records.append(
            {
                "stock_id": stock_id,
                "date": _day(row["date"], "TaiwanStockPrice"),
                **dict(
                    zip(("open", "high", "low", "close"), (p or None for p in prices), strict=True)
                ),
                "volume": _count(row["Trading_Volume"], "TaiwanStockPrice"),
                "turnover": _count(row["Trading_money"], "TaiwanStockPrice"),
            }
        )
    return pd.DataFrame(records, columns=PRICE_COLUMNS, dtype=object)


def _ex_factor(row: Row, stock_id: str) -> float | None:
    """除權息參考價 ÷ 前收; rights issues included, so it is not the dividend-only reference."""
    before, after = row["before_price"], row["after_price"]
    if not isinstance(before, int | float) or not isinstance(after, int | float):
        raise SchemaChangedError(f"FinMind dividend result for {stock_id} has bad prices")
    if before > 0 and after > 0:
        return float(after) / float(before)
    logger.warning("%s %s: no usable reference price, adj_factor left empty", stock_id, row["date"])
    return None


def dividend_events(
    policy: Iterable[Row],
    results: Iterable[Row],
    stock_id: str,
    start: date = date.min,
    end: date = date.max,
) -> pd.DataFrame:
    """One dividend_event per ex-date in range: amounts from the policy, factor from the result."""
    events: dict[date, dict[str, Any]] = {}

    def event(day: date) -> dict[str, Any]:
        return events.setdefault(day, {"cash": Decimal(0), "stock": Decimal(0), "factor": None})

    for row in _checked(policy, POLICY_KEYS, "TaiwanStockDividend", stock_id):
        for kind, ex_key, keys in DIVIDEND_PARTS:
            ex_date = _optional_day(row[ex_key], "TaiwanStockDividend")
            if ex_date is None:
                continue
            target = event(ex_date)
            target[kind] += sum(_decimal(row[key], "TaiwanStockDividend") for key in keys)
    for row in _checked(results, RESULT_KEYS, "TaiwanStockDividendResult", stock_id):
        event(_day(row["date"], "TaiwanStockDividendResult"))["factor"] = _ex_factor(row, stock_id)
    records = []
    for ex_date in sorted(day for day in events if start <= day <= end):
        found = events[ex_date]
        if found["factor"] is None:
            logger.warning("%s %s: no ex-right result, adj_factor left empty", stock_id, ex_date)
        records.append(
            {
                "ex_date": ex_date,
                "stock_id": stock_id,
                "cash_dividend": found["cash"].quantize(DIVIDEND_PLACES).normalize(),
                "stock_dividend": found["stock"].quantize(DIVIDEND_PLACES).normalize(),
                "adj_factor": found["factor"],
            }
        )
    return pd.DataFrame(records, columns=EVENT_COLUMNS, dtype=object)


def _number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaChangedError(f"FinMind {label} has a bad number {value!r}")
    return float(value)


def _positive(value: float | None) -> float | None:
    return value if value is not None and 0 < value < float("inf") else None


def valuations(rows: Iterable[Row], stock_id: str) -> pd.DataFrame:
    """TaiwanStockPER → daily_valuation; 0 means not published (losses), yield % → fraction."""
    records = []
    for row in _checked(rows, VALUATION_KEYS, "TaiwanStockPER", stock_id):
        dividend_yield = _number(row["dividend_yield"], "TaiwanStockPER")
        fraction = dividend_yield / PERCENT if dividend_yield is not None else None
        records.append(
            {
                "stock_id": stock_id,
                "date": _day(row["date"], "TaiwanStockPER"),
                "pe": _positive(_number(row["PER"], "TaiwanStockPER")),
                "pb": _positive(_number(row["PBR"], "TaiwanStockPER")),
                "dividend_yield": fraction if fraction is not None and 0 <= fraction < 1 else None,
            }
        )
    return pd.DataFrame(records, columns=VALUATION_COLUMNS, dtype=object)


def share_counts(rows: Iterable[Row], stock_id: str) -> pd.DataFrame:
    """TaiwanStockShareholding → shares_outstanding, one row per change of shares issued."""
    checked = _checked(rows, SHARE_KEYS, "TaiwanStockShareholding", stock_id)
    records: list[dict[str, Any]] = []
    for row in sorted(checked, key=lambda item: str(item["date"])):
        shares = row["NumberOfSharesIssued"]
        if shares is None:
            continue
        if isinstance(shares, bool) or not isinstance(shares, int):
            raise SchemaChangedError(f"FinMind TaiwanStockShareholding has a bad count {shares!r}")
        if shares <= 0:
            continue
        if records and records[-1]["shares"] == shares:
            continue
        records.append(
            {
                "stock_id": stock_id,
                "effective_date": _day(row["date"], "TaiwanStockShareholding"),
                "shares": shares,
                "reason": ShareChangeReason.UNKNOWN,
            }
        )
    return pd.DataFrame(records, columns=SHARE_COLUMNS, dtype=object)
