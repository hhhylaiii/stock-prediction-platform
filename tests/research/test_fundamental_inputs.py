import math
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from apps.market.models import DailyValuation, FinancialStatement, MonthlyRevenue
from apps.research.features.fundamental import attach_fundamentals
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2015, 1, 1), date(2016, 12, 31))
pytestmark = pytest.mark.django_db


def d(text: str) -> date:
    return date.fromisoformat(text)


def panel_of(stock_id: str, start: str, end: str, segment: int = 1, age: int = 500) -> pd.DataFrame:
    days = CALENDAR.between(d(start), d(end))
    return pd.DataFrame(
        {
            "stock_id": stock_id,
            "date": days,
            "segment": segment,
            "age": [age + k for k in range(len(days))],
        }
    )


def statement(period: str, announce: str, eps: float = 1.0, **fields: object) -> None:
    defaults: dict[str, object] = {
        "stock_id": "2330",
        "fin_type": "consolidated",
        "eps": Decimal(str(eps)),
        "gross_margin": 0.4,
        "op_margin": 0.2,
        "net_income": 100,
        "roe": 0.05,
        "debt_ratio": 0.3,
        "operating_cf": 120,
    }
    FinancialStatement.objects.create(
        period=period, announce_date=d(announce), **{**defaults, **fields}
    )


def year_of_statements(eps: float = 1.0) -> None:
    for period, announce in (
        ("2015-Q1", "2015-05-15"),
        ("2015-Q2", "2015-08-14"),
        ("2015-Q3", "2015-11-13"),
    ):
        statement(period, announce, eps)


@pytest.mark.spec8
def test_a_statement_is_invisible_before_its_announce_date() -> None:
    year_of_statements()
    statement("2015-Q4", "2016-03-31", eps=5.0)
    panel = panel_of("2330", "2016-03-28", "2016-04-01")

    attached = attach_fundamentals(panel, CALENDAR, d("2016-03-28"), d("2016-04-01"))
    by_day = attached.set_index("date")

    before = by_day.loc[: d("2016-03-30")]
    assert before["eps_ttm"].isna().all()
    assert (before["fund_gross_margin"] == 0.4).all()
    assert by_day["eps_ttm"].iloc[-2:].tolist() == [8.0, 8.0]
    assert by_day["fund_days_since_fs"].tolist() == [96.0, 97.0, 98.0, 0.0, 1.0]


def test_revenue_and_valuation_join_by_announcement_and_day() -> None:
    for k in range(14):
        year, month = divmod(2015 * 12 + k, 12)
        MonthlyRevenue.objects.create(
            stock_id="2330",
            period=f"{year}-{month + 1:02d}",
            revenue=100 + 10 * k,
            announce_date=date(year + (month + 1) // 12, (month + 1) % 12 + 1, 10),
        )
    DailyValuation.objects.create(
        stock_id="2330", date=d("2016-03-10"), pb=2.0, dividend_yield=0.04
    )
    panel = panel_of("2330", "2016-03-09", "2016-03-10")

    attached = attach_fundamentals(panel, CALENDAR, d("2016-03-09"), d("2016-03-10"))

    assert attached["fund_rev_yoy"].tolist() == pytest.approx([220 / 100 - 1, 230 / 110 - 1])
    assert math.isnan(attached["pb"].iloc[0])
    assert attached[["pb", "dividend_yield"]].iloc[1].tolist() == [2.0, 0.04]


def test_a_reused_code_does_not_inherit_the_old_companys_statements() -> None:
    year_of_statements()
    statement("2015-Q4", "2016-01-29")
    old = panel_of("2330", "2016-02-01", "2016-02-02", segment=1)
    new = panel_of("2330", "2016-12-28", "2016-12-30", segment=2, age=0)

    attached = attach_fundamentals(
        pd.concat([old, new], ignore_index=True), CALENDAR, d("2016-02-01"), d("2016-12-30")
    )

    assert attached["eps_ttm"].tolist()[:2] == [4.0, 4.0]
    assert attached["eps_ttm"].iloc[2:].isna().all()
    assert attached["fund_days_since_fs"].iloc[2:].isna().all()


def test_a_weekend_announcement_counts_from_the_next_trading_day() -> None:
    year_of_statements()
    statement("2015-Q4", "2016-03-26")
    panel = panel_of("2330", "2016-03-25", "2016-03-29")

    attached = attach_fundamentals(panel, CALENDAR, d("2016-03-25"), d("2016-03-29"))

    assert attached["fund_days_since_fs"].tolist()[1:] == [1.0, 2.0]
