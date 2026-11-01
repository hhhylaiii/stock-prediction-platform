from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import pytest

from apps.etl.adapters.finmind_fundamentals import monthly_revenues, statements
from apps.etl.errors import SchemaChangedError
from apps.market.calendar import TradingDays

Rows = list[dict[str, Any]]
WEEKDAYS = TradingDays(
    (
        date(2010, 1, 1) + timedelta(days=n)
        for n in range(6200)
        if (date(2010, 1, 1) + timedelta(days=n)).weekday() < 5
    ),
    date(2010, 1, 1),
    date(2026, 12, 31),
)
YEAR_2024 = (date(2024, 1, 1), date(2024, 12, 31))


@pytest.fixture
def rows(load_fixture: Callable[[str], Any]) -> Callable[[str], Rows]:
    return lambda name: list(load_fixture(f"finmind/{name}.json")["data"])


def _record(frame: Any, period: str) -> dict[str, Any]:
    return dict(frame.loc[frame["period"] == period].iloc[0])


def _revenues(revenue: Rows) -> Any:
    return monthly_revenues(revenue, "2330", WEEKDAYS, *YEAR_2024)


def test_revenue_periods_come_from_the_reported_month(rows: Callable[[str], Rows]) -> None:
    frame = _revenues(rows("revenue_2330_2023_2024"))
    assert frame["period"].tolist() == ["2023-12", *(f"2024-{m:02d}" for m in range(1, 12))]
    january = _record(frame, "2024-01")
    assert january["revenue"] == 215_785_127_000
    assert january["yoy"] == pytest.approx(215_785_127_000 / 200_050_544_000 - 1)
    assert january["mom"] == pytest.approx(215_785_127_000 / 176_299_866_000 - 1)
    assert january["announce_date"] == date(2024, 2, 13)


def test_growth_needs_a_positive_earlier_month(rows: Callable[[str], Rows]) -> None:
    revenue = [
        {**row, "revenue": 0} if row["date"] == "2023-02-01" else row
        for row in rows("revenue_2330_2023_2024")
        if row["date"] != "2023-12-01"
    ]
    frame = _revenues(revenue)
    assert _record(frame, "2024-01")["yoy"] is None
    assert _record(frame, "2023-12")["mom"] is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"revenue": 1.5}, "bad revenue"),
        ({"revenue_month": 13}, "bad period"),
        ({"revenue_year": "2024"}, "bad period"),
    ],
)
def test_bad_revenue_rows_are_schema_changes(
    rows: Callable[[str], Rows], change: dict[str, Any], message: str
) -> None:
    revenue = rows("revenue_2330_2023_2024")
    with pytest.raises(SchemaChangedError, match=message):
        _revenues([*revenue[:-1], {**revenue[-1], **change}])


def test_a_repeated_revenue_month_is_a_schema_change(rows: Callable[[str], Rows]) -> None:
    revenue = rows("revenue_2330_2023_2024")
    with pytest.raises(SchemaChangedError, match="repeated 2024-11"):
        _revenues([*revenue, revenue[-1]])


def test_revenue_rows_without_a_key_are_schema_changes(rows: Callable[[str], Rows]) -> None:
    revenue = [
        {k: v for k, v in r.items() if k != "revenue_month"} for r in rows("revenue_2330_2023_2024")
    ]
    with pytest.raises(SchemaChangedError, match="revenue_month"):
        _revenues(revenue)


def _statements(rows: Callable[[str], Rows], stock: str, span: str, start: date, end: date) -> Any:
    return statements(
        rows(f"income_{stock}_{span}"),
        rows(f"balance_{stock}_{span}"),
        rows(f"cashflow_{stock}_{span}") if stock == "2330" else [],
        stock,
        WEEKDAYS,
        start,
        end,
    )


def test_ifrs_statements_become_single_quarter_ratios(rows: Callable[[str], Rows]) -> None:
    frame = _statements(rows, "2330", "2023_2024", *YEAR_2024)
    assert frame["period"].tolist() == ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"]
    q1 = _record(frame, "2024-Q1")
    assert q1["fin_type"] == "consolidated"
    assert q1["eps"] == Decimal("8.7")
    assert q1["revenue"] == 592_644_201_000
    assert q1["gross_margin"] == pytest.approx(314_505_269_000 / 592_644_201_000)
    assert q1["op_margin"] == pytest.approx(249_018_306_000 / 592_644_201_000)
    assert q1["net_income"] == 225_484_877_000
    assert q1["debt_ratio"] == pytest.approx(2_122_174_828_000 / 5_787_891_082_000)
    average_equity = (3_635_731_930_000 + 3_458_913_627_000) / 2
    assert q1["roe"] == pytest.approx(225_484_877_000 / average_equity)
    assert q1["operating_cf"] == 436_311_108_000
    assert q1["announce_date"] == date(2024, 5, 31)
    assert _record(frame, "2024-Q2")["operating_cf"] == 813_979_318_000 - 436_311_108_000
    q4 = _record(frame, "2024-Q4")
    assert q4["revenue"] == 868_461_178_000
    assert q4["announce_date"] == date(2025, 4, 1)


def test_financial_companies_have_no_margins_and_equity_excludes_minorities(
    rows: Callable[[str], Rows],
) -> None:
    q1 = _record(_statements(rows, "2884", "2023_2024", *YEAR_2024), "2024-Q1")
    assert (q1["gross_margin"], q1["op_margin"], q1["operating_cf"]) == (None, None, None)
    assert q1["net_income"] == 6_250_969_000
    assert q1["debt_ratio"] == pytest.approx(3_489_129_204_000 / 3_734_752_361_000)
    average_equity = ((245_623_157_000 - 179_643_000) + (236_423_847_000 - 176_167_000)) / 2
    assert q1["roe"] == pytest.approx(6_250_969_000 / average_equity)


def test_statements_before_2013_are_individual_and_income_only(
    rows: Callable[[str], Rows],
) -> None:
    frame = _statements(rows, "2330", "2010_2013", date(2012, 1, 1), date(2012, 12, 31))
    q4 = _record(frame, "2012-Q4")
    assert q4["fin_type"] == "individual"
    assert (q4["revenue"], q4["net_income"], q4["eps"]) == (
        129_485_621_000,
        41_569_268_000,
        Decimal("1.6"),
    )
    assert (q4["debt_ratio"], q4["roe"], q4["operating_cf"]) == (None, None, None)
    assert q4["announce_date"] == date(2013, 4, 2)


def test_a_missing_previous_quarter_leaves_flows_and_roe_empty(
    rows: Callable[[str], Rows],
) -> None:
    frame = statements(
        rows("income_2330_2023_2024"),
        [r for r in rows("balance_2330_2023_2024") if r["date"] != "2024-03-31"],
        [r for r in rows("cashflow_2330_2023_2024") if r["date"] != "2024-03-31"],
        "2330",
        WEEKDAYS,
        *YEAR_2024,
    )
    q2 = _record(frame, "2024-Q2")
    assert (q2["roe"], q2["operating_cf"]) == (None, None)
    assert _record(frame, "2024-Q1")["debt_ratio"] is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"date": "2024-05-31"}, "not a quarter end"),
        ({"value": "1,000"}, "bad value"),
    ],
)
def test_bad_statement_rows_are_schema_changes(
    rows: Callable[[str], Rows], change: dict[str, Any], message: str
) -> None:
    income = rows("income_2330_2023_2024")
    with pytest.raises(SchemaChangedError, match=message):
        statements([*income, {**income[0], **change}], [], [], "2330", WEEKDAYS, *YEAR_2024)


def test_a_repeated_line_item_is_a_schema_change(rows: Callable[[str], Rows]) -> None:
    income = rows("income_2330_2023_2024")
    revenue = next(r for r in income if r["type"] == "Revenue")
    with pytest.raises(SchemaChangedError, match="repeated Revenue"):
        statements([*income, revenue], [], [], "2330", WEEKDAYS, *YEAR_2024)


def test_statement_rows_without_a_key_are_schema_changes(rows: Callable[[str], Rows]) -> None:
    income = [{k: v for k, v in r.items() if k != "type"} for r in rows("income_2330_2023_2024")]
    with pytest.raises(SchemaChangedError, match="type"):
        statements(income, [], [], "2330", WEEKDAYS, *YEAR_2024)


def test_balance_sheets_alone_before_2013_give_no_row(rows: Callable[[str], Rows]) -> None:
    balance = rows("balance_2330_2010_2013")
    frame = statements([], balance, [], "2330", WEEKDAYS, date(2012, 1, 1), date(2012, 12, 31))
    assert frame.empty


def test_an_empty_value_counts_as_missing(rows: Callable[[str], Rows]) -> None:
    income = [
        {**row, "value": None} if row["type"] == "Revenue" else row
        for row in rows("income_2330_2023_2024")
    ]
    frame = statements(income, [], [], "2330", WEEKDAYS, *YEAR_2024)
    q1 = _record(frame, "2024-Q1")
    assert (q1["revenue"], q1["gross_margin"]) == (None, None)
    assert q1["net_income"] == 225_484_877_000
