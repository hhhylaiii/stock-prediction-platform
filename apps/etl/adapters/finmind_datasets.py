"""FinMind rows mapped onto the standard tables; units checked against TWSE (SPEC §4)."""

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from apps.etl.dividend_dates import Session
from apps.etl.errors import SchemaChangedError
from apps.market.models import ShareChangeReason

logger = logging.getLogger(__name__)
Row = Mapping[str, Any]
TAIPEI = ZoneInfo("Asia/Taipei")
DAY_END = time(23, 59, 59)
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
    "announce_date",
]
VALUATION_KEYS = ("date", "stock_id", "PER", "PBR", "dividend_yield")
VALUATION_COLUMNS = ["stock_id", "date", "pe", "pb", "dividend_yield"]
SHARE_KEYS = ("date", "stock_id", "NumberOfSharesIssued")
SHARE_COLUMNS = ["stock_id", "effective_date", "shares", "reason"]
DIVIDEND_PARTS = (
    ("stock", "StockExDividendTradingDate", ("StockEarningsDistribution", "StockStatutorySurplus")),
    ("cash", "CashExDividendTradingDate", ("CashEarningsDistribution", "CashStatutorySurplus")),
)
INSTITUTIONAL_KEYS = ("date", "stock_id", "name", "buy", "sell")
INVESTOR_COLUMNS = {
    "Foreign_Investor": "fi_net",
    "Foreign_Dealer_Self": "fi_net",
    "Investment_Trust": "it_net",
    "Dealer_self": "dealer_self_net",
    "Dealer_Hedging": "dealer_hedge_net",
    "Dealer": None,
}
LOTS = {
    "MarginPurchaseTodayBalance": "margin_balance",
    "MarginPurchaseLimit": "margin_limit",
    "ShortSaleTodayBalance": "short_balance",
}
MARGIN_KEYS = ("date", "stock_id", *LOTS)
LENDING_KEYS = (
    "date",
    "stock_id",
    "SBLShortSalesCurrentDayBalance",
    "MarginShortSalesCurrentDayBalance",
)
LENDING_SHORT = "lending_short_balance"
HOLDING_KEYS = ("date", "stock_id", "ForeignInvestmentSharesRatio")
CHIP_COLUMNS = [
    "stock_id",
    "date",
    "fi_net",
    "it_net",
    "dealer_self_net",
    "dealer_hedge_net",
    "margin_balance",
    "margin_limit",
    "short_balance",
    "sbl_short_balance",
    "fi_holding_pct",
]
SHARES_PER_LOT = 1000
DIVIDEND_PLACES = Decimal("0.00000001")
PERCENT = 100.0


def checked_rows(rows: Iterable[Row], keys: Sequence[str], label: str, stock_id: str) -> list[Row]:
    checked = []
    for row in rows:
        missing = [key for key in keys if key not in row]
        if missing:
            raise SchemaChangedError(f"FinMind {label} rows lost keys {missing}")
        if str(row["stock_id"]).strip() != stock_id:
            raise SchemaChangedError(f"FinMind {label} returned {row['stock_id']!r} for {stock_id}")
        checked.append(row)
    return checked


def parse_day(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        raise SchemaChangedError(f"FinMind {label} has a bad date {value!r}") from None


def _optional_day(value: Any, label: str) -> date | None:
    return parse_day(value, label) if str(value or "").strip() else None


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
    for row in checked_rows(rows, PRICE_KEYS, "TaiwanStockPrice", stock_id):
        prices = [_decimal(row[key], "TaiwanStockPrice") for key in ("open", "max", "min", "close")]
        records.append(
            {
                "stock_id": stock_id,
                "date": parse_day(row["date"], "TaiwanStockPrice"),
                **dict(
                    zip(("open", "high", "low", "close"), (p or None for p in prices), strict=True)
                ),
                "volume": _count(row["Trading_Volume"], "TaiwanStockPrice"),
                "turnover": _count(row["Trading_money"], "TaiwanStockPrice"),
            }
        )
    _require_unique(records, "date", "TaiwanStockPrice")
    return pd.DataFrame(records, columns=PRICE_COLUMNS, dtype=object)


def _require_unique(records: list[dict[str, Any]], key: str, label: str) -> None:
    """A day sent twice would otherwise fail later in the upsert, stopping the whole batch."""
    seen: set[Any] = set()
    for record in records:
        if record[key] in seen:
            raise SchemaChangedError(f"FinMind {label} repeated {record[key]}")
        seen.add(record[key])


def _ratio(row: Row, after_key: str, before_key: str, label: str) -> float | None:
    """Reference price ÷ last close; for dividends this is 除權息參考價, rights issues included."""
    before, after = row[before_key], row[after_key]
    if not isinstance(before, int | float) or not isinstance(after, int | float):
        raise SchemaChangedError(f"FinMind {label} for {row['stock_id']} has bad prices")
    if before > 0 and after > 0:
        return float(after) / float(before)
    logger.warning("%s %s: %s has no usable reference price", row["stock_id"], row["date"], label)
    return None


def ex_date_announcements(
    policy: Iterable[Row], stock_id: str, session: Session
) -> dict[date, date | None]:
    """The session each ex-date was first announced in; none if that comes after the ex-date."""
    seen: dict[date, date] = {}
    keys = (*POLICY_KEYS, "AnnouncementDate")
    for row in checked_rows(policy, keys, "TaiwanStockDividend", stock_id):
        day = _optional_day(row.get("AnnouncementDate"), "TaiwanStockDividend")
        ex_dates = [_optional_day(row[key], "TaiwanStockDividend") for _, key, _ in DIVIDEND_PARTS]
        if day is None or not any(ex_dates):
            continue
        moment = session(datetime.combine(day, _clock(row.get("AnnouncementTime")), TAIPEI))
        for ex_date in ex_dates:
            if ex_date is not None:
                seen[ex_date] = min(seen.get(ex_date, moment), moment)
    late = sorted(ex for ex, moment in seen.items() if moment > ex)
    if late:
        logger.warning("%s: announced after the ex-date %s, left without a date", stock_id, late)
    return {ex: (None if moment > ex else moment) for ex, moment in seen.items()}


def _clock(value: Any) -> time:
    """A missing announcement time counts as the end of the day."""
    text = str(value or "").strip()
    if not text:
        return DAY_END
    try:
        return time.fromisoformat(text)
    except ValueError:
        raise SchemaChangedError(f"FinMind TaiwanStockDividend has a bad time {value!r}") from None


def dividend_events(
    policy: Iterable[Row],
    results: Iterable[Row],
    stock_id: str,
    start: date = date.min,
    end: date = date.max,
    reductions: Iterable[Row] = (),
    splits: Iterable[Row] = (),
    session: Session | None = None,
) -> pd.DataFrame:
    """One event per ex-date in range; capital reductions and splits add factor-only events."""
    policy = list(policy)
    announced = {} if session is None else ex_date_announcements(policy, stock_id, session)
    events: dict[date, dict[str, Any]] = {}

    def event(day: date) -> dict[str, Any]:
        return events.setdefault(day, {"cash": Decimal(0), "stock": Decimal(0), "factors": {}})

    for row in checked_rows(policy, POLICY_KEYS, "TaiwanStockDividend", stock_id):
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
        for row in checked_rows(rows, required, label, stock_id):
            day = parse_day(row["date"], label)
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
                "announce_date": announced.get(ex_date),
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
    for row in checked_rows(rows, VALUATION_KEYS, "TaiwanStockPER", stock_id):
        dividend_yield = _number(row["dividend_yield"], "TaiwanStockPER")
        fraction = dividend_yield / PERCENT if dividend_yield is not None else None
        records.append(
            {
                "stock_id": stock_id,
                "date": parse_day(row["date"], "TaiwanStockPER"),
                "pe": _positive(_number(row["PER"], "TaiwanStockPER")),
                "pb": _positive(_number(row["PBR"], "TaiwanStockPER")),
                "dividend_yield": fraction if fraction is not None and 0 <= fraction < 1 else None,
            }
        )
    _require_unique(records, "date", "TaiwanStockPER")
    return pd.DataFrame(records, columns=VALUATION_COLUMNS, dtype=object)


def share_counts(rows: Iterable[Row], stock_id: str) -> pd.DataFrame:
    """TaiwanStockShareholding → shares_outstanding, one row per change of shares issued."""
    checked = checked_rows(rows, SHARE_KEYS, "TaiwanStockShareholding", stock_id)
    days = [{"date": parse_day(row["date"], "TaiwanStockShareholding")} for row in checked]
    _require_unique(days, "date", "TaiwanStockShareholding")
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
                "effective_date": parse_day(row["date"], "TaiwanStockShareholding"),
                "shares": shares,
                "reason": ShareChangeReason.UNKNOWN,
            }
        )
    return pd.DataFrame(records, columns=SHARE_COLUMNS, dtype=object)


def _signed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaChangedError(f"FinMind {label} has a bad count {value!r}")
    return value


def chip_rows(
    institutional: Iterable[Row],
    margin: Iterable[Row],
    lending: Iterable[Row],
    holding: Iterable[Row],
    stock_id: str,
) -> pd.DataFrame:
    """Four FinMind datasets joined into daily_chip rows; margin lots become shares."""
    days: dict[date, dict[str, Any]] = {}
    starts = _fill_investors(days, institutional, stock_id)
    parts: list[tuple[Iterable[Row], Sequence[str], str, Callable[[Row, str], dict[str, Any]]]] = [
        (margin, MARGIN_KEYS, "TaiwanStockMarginPurchaseShortSale", _lots),
        (lending, LENDING_KEYS, "TaiwanDailyShortSaleBalances", _lending),
        (holding, HOLDING_KEYS, "TaiwanStockShareholding", _holding),
    ]
    for rows, required, label, columns in parts:
        filled: set[date] = set()
        for row in checked_rows(rows, required, label, stock_id):
            day = parse_day(row["date"], label)
            if day in filled:
                raise SchemaChangedError(f"FinMind {label} repeated {day}")
            filled.add(day)
            days.setdefault(day, {}).update(columns(row, label))
    _drop_misaligned_lending(days, stock_id)
    _zero_quiet_days(days, starts, stock_id)
    records = [{"stock_id": stock_id, "date": day, **days[day]} for day in sorted(days)]
    frame = pd.DataFrame(records, columns=CHIP_COLUMNS, dtype=object)
    return frame.astype(object).where(frame.notna(), None)


def _fill_investors(
    days: dict[date, dict[str, Any]], rows: Iterable[Row], stock_id: str
) -> dict[str, date]:
    """Net shares per investor column; returns each column's first day in this response."""
    label = "TaiwanStockInstitutionalInvestorsBuySell"
    seen: set[tuple[date, str]] = set()
    starts: dict[str, date] = {}
    for row in checked_rows(rows, INSTITUTIONAL_KEYS, label, stock_id):
        name = str(row["name"])
        if name not in INVESTOR_COLUMNS:
            raise SchemaChangedError(f"FinMind {label} has an unknown investor type {name!r}")
        day = parse_day(row["date"], label)
        if (day, name) in seen:
            raise SchemaChangedError(f"FinMind {label} repeated {name} on {day}")
        seen.add((day, name))
        column = INVESTOR_COLUMNS[name]
        if column is not None:
            starts[column] = min(starts.get(column, day), day)
            target = days.setdefault(day, {})
            net = _signed(row["buy"], label) - _signed(row["sell"], label)
            target[column] = target.get(column, 0) + net
    return starts


def _zero_quiet_days(
    days: dict[date, dict[str, Any]], starts: Mapping[str, date], stock_id: str
) -> None:
    """T86 omits a stock with no institutional trades that day; such days net 0, not unknown."""
    quiet = 0
    for day, columns in days.items():
        missing = [name for name, first in starts.items() if day >= first and name not in columns]
        quiet += bool(missing)
        columns.update(dict.fromkeys(missing, 0))
    if quiet:
        logger.info("%s: %d day without institutional rows, nets set to 0", stock_id, quiet)


def _balance(value: Any, label: str) -> int:
    count = _signed(value, label)
    if count < 0:
        raise SchemaChangedError(f"FinMind {label} has a bad count {value!r}")
    return count


def _lots(row: Row, label: str) -> dict[str, Any]:
    return {column: _balance(row[key], label) * SHARES_PER_LOT for key, column in LOTS.items()}


def _lending(row: Row, label: str) -> dict[str, Any]:
    return {
        "sbl_short_balance": _balance(row["SBLShortSalesCurrentDayBalance"], label),
        LENDING_SHORT: _balance(row["MarginShortSalesCurrentDayBalance"], label),
    }


def _drop_misaligned_lending(days: dict[date, dict[str, Any]], stock_id: str) -> None:
    """A lending row whose short balance disagrees with the margin dataset is misaligned."""
    for day, columns in days.items():
        short = columns.pop(LENDING_SHORT, None)
        if short is not None and columns.get("short_balance") not in (None, short):
            logger.warning(
                "%s %s: lending row misaligned, sbl_short_balance dropped", stock_id, day
            )
            columns["sbl_short_balance"] = None


def _holding(row: Row, label: str) -> dict[str, Any]:
    ratio = _number(row["ForeignInvestmentSharesRatio"], label)
    if ratio is None or not 0 <= ratio <= PERCENT:
        logger.warning("%s %s: foreign holding %s dropped", row["stock_id"], row["date"], ratio)
        ratio = None
    return {"fi_holding_pct": ratio}
