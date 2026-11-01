"""FinMind monthly revenue and financial statements mapped onto the standard tables (SPEC §4)."""

import math
from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd

from apps.etl.adapters.finmind_datasets import Row, checked_rows, parse_day
from apps.etl.deadlines import revenue_deadline, statement_deadline, visible_from
from apps.etl.errors import SchemaChangedError
from apps.market.calendar import TradingDays
from apps.market.models import FinType

REVENUE_LABEL = "TaiwanStockMonthRevenue"
REVENUE_KEYS = ("date", "stock_id", "revenue", "revenue_month", "revenue_year")
REVENUE_COLUMNS = ["stock_id", "period", "revenue", "yoy", "mom", "announce_date"]
STATEMENT_KEYS = ("date", "stock_id", "type", "value")
VALUE_COLUMNS = [
    "eps",
    "revenue",
    "gross_margin",
    "op_margin",
    "net_income",
    "roe",
    "debt_ratio",
    "operating_cf",
]
STATEMENT_COLUMNS = ["stock_id", "period", "fin_type", *VALUE_COLUMNS, "announce_date"]
NET_INCOME_TYPES = ("EquityAttributableToOwnersOfParent", "NetIncome")
OPERATING_CF_TYPES = ("CashFlowsFromOperatingActivities", "NetCashInflowFromOperatingActivities")
USED_TYPES = frozenset(
    {
        "EPS",
        "Revenue",
        "GrossProfit",
        "OperatingIncome",
        *NET_INCOME_TYPES,
        "TotalAssets",
        "Liabilities",
        "Equity",
        "NoncontrollingInterests",
        *OPERATING_CF_TYPES,
    }
)
QUARTER_ENDS = {(3, 31): 1, (6, 30): 2, (9, 30): 3, (12, 31): 4}
IFRS_FROM = 2013
EPS_PLACES = Decimal("0.0001")
MONTHS = 12
Sheet = Mapping[date, Mapping[str, float]]


def monthly_revenues(
    rows: Iterable[Row], stock_id: str, calendar: TradingDays, start: date, end: date
) -> pd.DataFrame:
    """Months published in [start, end]; growth may use months fetched before start."""
    found: dict[str, tuple[date, int]] = {}
    for row in checked_rows(rows, REVENUE_KEYS, REVENUE_LABEL, stock_id):
        period = _month_period(row["revenue_year"], row["revenue_month"])
        revenue = row["revenue"]
        if isinstance(revenue, bool) or not isinstance(revenue, int):
            raise SchemaChangedError(f"FinMind {REVENUE_LABEL} has a bad revenue {revenue!r}")
        if period in found:
            raise SchemaChangedError(f"FinMind {REVENUE_LABEL} repeated {period}")
        found[period] = (parse_day(row["date"], REVENUE_LABEL), revenue)
    records = [
        {
            "stock_id": stock_id,
            "period": period,
            "revenue": revenue,
            "yoy": _growth(revenue, found.get(_shift_month(period, -MONTHS))),
            "mom": _growth(revenue, found.get(_shift_month(period, -1))),
            "announce_date": visible_from(revenue_deadline(period), calendar),
        }
        for period, (published, revenue) in sorted(found.items())
        if start <= published <= end
    ]
    return pd.DataFrame(records, columns=REVENUE_COLUMNS, dtype=object)


def _month_period(year: Any, month: Any) -> str:
    numbers = [value for value in (year, month) if type(value) is int]
    if len(numbers) < 2 or not 1 <= month <= MONTHS:
        raise SchemaChangedError(f"FinMind {REVENUE_LABEL} has a bad period {year!r}-{month!r}")
    return f"{year:04d}-{month:02d}"


def _shift_month(period: str, months: int) -> str:
    index = int(period[:4]) * MONTHS + int(period[5:]) - 1 + months
    return f"{index // MONTHS:04d}-{index % MONTHS + 1:02d}"


def _growth(revenue: int, earlier: tuple[date, int] | None) -> float | None:
    if earlier is None or earlier[1] <= 0:
        return None
    return revenue / earlier[1] - 1


def statements(
    income: Iterable[Row],
    balance: Iterable[Row],
    cash_flow: Iterable[Row],
    stock_id: str,
    calendar: TradingDays,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Quarters ending in [start, end]; before 2013 only the individual income statement is used."""
    income_items = _pivot(income, "TaiwanStockFinancialStatements", stock_id)
    balance_items = _pivot(balance, "TaiwanStockBalanceSheet", stock_id)
    cash_items = _pivot(cash_flow, "TaiwanStockCashFlowsStatement", stock_id)
    records = []
    for day in sorted(income_items.keys() | balance_items.keys() | cash_items.keys()):
        if not start <= day <= end:
            continue
        individual = day.year < IFRS_FROM
        values = dict.fromkeys(VALUE_COLUMNS) | _income_values(income_items.get(day, {}))
        if not individual:
            values |= _balance_values(day, values["net_income"], balance_items, cash_items)
        if all(value is None for value in values.values()):
            continue
        period = f"{day.year}-Q{QUARTER_ENDS[(day.month, day.day)]}"
        records.append(
            {
                "stock_id": stock_id,
                "period": period,
                "fin_type": FinType.INDIVIDUAL if individual else FinType.CONSOLIDATED,
                **values,
                "announce_date": visible_from(statement_deadline(period), calendar),
            }
        )
    return pd.DataFrame(records, columns=STATEMENT_COLUMNS, dtype=object)


def _pivot(rows: Iterable[Row], label: str, stock_id: str) -> dict[date, dict[str, float]]:
    items: dict[date, dict[str, float]] = {}
    for row in checked_rows(rows, STATEMENT_KEYS, label, stock_id):
        day = parse_day(row["date"], label)
        if (day.month, day.day) not in QUARTER_ENDS:
            raise SchemaChangedError(f"FinMind {label} date {day} is not a quarter end")
        kind, value = str(row["type"]), row["value"]
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise SchemaChangedError(f"FinMind {label} has a bad value {value!r} for {kind}")
        sheet = items.setdefault(day, {})
        if kind in USED_TYPES and kind in sheet:
            raise SchemaChangedError(f"FinMind {label} repeated {kind} on {day}")
        sheet[kind] = float(value)
    return items


def _income_values(items: Mapping[str, float]) -> dict[str, Any]:
    revenue = items.get("Revenue")
    eps = items.get("EPS")
    return {
        "eps": None if eps is None else Decimal(str(eps)).quantize(EPS_PLACES),
        "revenue": _whole(revenue),
        "gross_margin": _share(items.get("GrossProfit"), revenue),
        "op_margin": _share(items.get("OperatingIncome"), revenue),
        "net_income": _whole(_first(items, NET_INCOME_TYPES)),
    }


def _balance_values(
    day: date, net_income: int | None, balance: Sheet, cash: Sheet
) -> dict[str, Any]:
    sheet = balance.get(day, {})
    previous = _previous_quarter_end(day)
    now, before = _parent_equity(sheet), _parent_equity(balance.get(previous, {}))
    roe = None
    if net_income is not None and now is not None and before is not None and now + before > 0:
        roe = net_income / ((now + before) / 2)
    flow = _first(cash.get(day, {}), OPERATING_CF_TYPES)
    earlier = 0.0 if day.month == 3 else _first(cash.get(previous, {}), OPERATING_CF_TYPES)
    return {
        "roe": roe,
        "debt_ratio": _share(sheet.get("Liabilities"), sheet.get("TotalAssets")),
        "operating_cf": None if flow is None or earlier is None else _whole(flow - earlier),
    }


def _previous_quarter_end(day: date) -> date:
    if day.month == 3:
        return date(day.year - 1, 12, 31)
    return next(date(day.year, m, d) for (m, d), q in QUARTER_ENDS.items() if m == day.month - 3)


def _parent_equity(sheet: Mapping[str, float]) -> float | None:
    if "EquityAttributableToOwnersOfParent" in sheet:
        return sheet["EquityAttributableToOwnersOfParent"]
    if "Equity" in sheet:
        return sheet["Equity"] - sheet.get("NoncontrollingInterests", 0.0)
    return None


def _first(items: Mapping[str, float], kinds: Iterable[str]) -> float | None:
    return next((items[kind] for kind in kinds if kind in items), None)


def _share(part: float | None, whole: float | None) -> float | None:
    return None if part is None or whole is None or whole <= 0 else part / whole


def _whole(value: float | None) -> int | None:
    return None if value is None else round(value)
