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
REDUCTION_KEYS = (
    "date",
    "stock_id",
    "ClosingPriceonTheLastTradingDay",
    "PostReductionReferencePrice",
)
SPLIT_KEYS = ("date", "stock_id", "before_price", "after_price")
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


def _ratio(row: Row, after_key: str, before_key: str, label: str) -> float | None:
    """Reference price ÷ last close; for dividends this is 除權息參考價, rights issues included."""
    before, after = row[before_key], row[after_key]
    if not isinstance(before, int | float) or not isinstance(after, int | float):
        raise SchemaChangedError(f"FinMind {label} for {row['stock_id']} has bad prices")
    if before > 0 and after > 0:
        return float(after) / float(before)
    logger.warning("%s %s: %s has no usable reference price", row["stock_id"], row["date"], label)
    return None


def dividend_events(
    policy: Iterable[Row],
    results: Iterable[Row],
    stock_id: str,
    start: date = date.min,
    end: date = date.max,
    reductions: Iterable[Row] = (),
    splits: Iterable[Row] = (),
) -> pd.DataFrame:
    """One event per ex-date in range; capital reductions and splits add factor-only events."""
    events: dict[date, dict[str, Any]] = {}

    def event(day: date) -> dict[str, Any]:
        return events.setdefault(day, {"cash": Decimal(0), "stock": Decimal(0), "factors": {}})

    for row in _checked(policy, POLICY_KEYS, "TaiwanStockDividend", stock_id):
        for kind, ex_key, keys in DIVIDEND_PARTS:
            ex_date = _optional_day(row[ex_key], "TaiwanStockDividend")
            if ex_date is None:
                continue
            target = event(ex_date)
            target[kind] += sum(_decimal(row[key], "TaiwanStockDividend") for key in keys)
    sources: list[tuple[Iterable[Row], Sequence[str], str, str, str]] = [
        (results, RESULT_KEYS, "TaiwanStockDividendResult", "after_price", "before_price"),
        (
            reductions,
            REDUCTION_KEYS,
            "TaiwanStockCapitalReductionReferencePrice",
            "PostReductionReferencePrice",
            "ClosingPriceonTheLastTradingDay",
        ),
        (splits, SPLIT_KEYS, "TaiwanStockSplitPrice", "after_price", "before_price"),
    ]
    for rows, required, label, after_key, before_key in sources:
        for row in _checked(rows, required, label, stock_id):
            day = _day(row["date"], label)
            factors = event(day)["factors"]
            if label in factors:
                logger.warning("%s %s: repeated %s rows, keeping the last", stock_id, day, label)
            factors[label] = _ratio(row, after_key, before_key, label)
    records = []
    for ex_date in sorted(day for day in events if start <= day <= end):
        found = events[ex_date]
        pays = found["cash"] > 0 or found["stock"] > 0
        if pays and "TaiwanStockDividendResult" not in found["factors"]:
            logger.warning("%s %s: no ex-right result, dividend not adjusted", stock_id, ex_date)
        factor = _combined(found["factors"], stock_id, ex_date)
        records.append(
            {
                "ex_date": ex_date,
                "stock_id": stock_id,
                "cash_dividend": found["cash"].quantize(DIVIDEND_PLACES).normalize(),
                "stock_dividend": found["stock"].quantize(DIVIDEND_PLACES).normalize(),
                "adj_factor": factor,
            }
        )
    return pd.DataFrame(records, columns=EVENT_COLUMNS, dtype=object)


def _combined(factors: Mapping[str, float | None], stock_id: str, ex_date: date) -> float | None:
    """Product of the usable factors of one day; None when no source gave one."""
    usable = [factor for factor in factors.values() if factor is not None]
    if len(factors) > 1:
        logger.warning("%s %s: %s on one day", stock_id, ex_date, ", ".join(sorted(factors)))
    if not usable:
        return None
    product = 1.0
    for factor in usable:
        product *= factor
    return product


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
