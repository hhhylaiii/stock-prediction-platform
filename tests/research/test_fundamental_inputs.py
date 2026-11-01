import math
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from apps.market.models import DailyValuation, FinancialStatement, MonthlyRevenue
from apps.research.features.fundamental import attach_fundamentals
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2013, 1, 1), date(2016, 12, 31))
REVENUE_NAMES = ["fund_rev_yoy", "fund_rev_3m_yoy", "fund_rev_mom"]
STATEMENT_NAMES = [
    "eps_ttm",
    "fund_eps_yoy",
    "fund_roe_ttm",
    "fund_gross_margin",
    "fund_op_margin",
    "fund_debt_ratio",
    "fund_gm_yoy_chg",
    "fund_opm_yoy_chg",
    "fund_cfo_to_ni",
]
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


def revenue(stock_id: str, first: str, values: list[float]) -> None:
    year, month = map(int, first.split("-"))
    for k, value in enumerate(values):
        y, m = divmod(year * 12 + month - 1 + k, 12)
        after_y, after_m = divmod(y * 12 + m + 1, 12)
        MonthlyRevenue.objects.create(
            stock_id=stock_id,
            period=f"{y}-{m + 1:02d}",
            revenue=value,
            announce_date=date(after_y, after_m + 1, 10),
        )


def attach(panel: pd.DataFrame, start: str | None = None) -> pd.DataFrame:
    first = d(start) if start else panel["date"].min()
    return attach_fundamentals(panel, CALENDAR, first, panel["date"].max())


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
    statement("2015-Q4", "2016-03-31", eps=5.0, gross_margin=0.5)
    panel = panel_of("2330", "2016-03-28", "2016-04-01")

    attached = attach_fundamentals(panel, CALENDAR, d("2016-03-28"), d("2016-04-01"))
    by_day = attached.set_index("date")

    before = by_day.loc[: d("2016-03-30")]
    assert before["eps_ttm"].isna().all()
    assert (before["fund_gross_margin"] == 0.4).all()
    assert by_day["fund_gross_margin"].iloc[-2:].tolist() == [0.5, 0.5]
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


def test_a_reused_code_reads_only_filings_of_the_new_listing() -> None:
    year_of_statements()
    statement("2015-Q4", "2016-01-29")
    statement("2016-Q1", "2016-05-13", eps=3.0, gross_margin=0.5)
    revenue("2330", "2015-01", [100.0] * 12)
    revenue("2330", "2016-03", [300.0, 330.0])
    old = panel_of("2330", "2016-02-01", "2016-02-02", segment=1)
    new = panel_of("2330", "2016-03-01", "2016-05-16", segment=2, age=0)

    attached = attach(pd.concat([old, new], ignore_index=True)).set_index("date")

    assert attached["eps_ttm"].iloc[:2].tolist() == [4.0, 4.0]
    before, after = attached.loc[d("2016-03-01") : d("2016-05-12")], attached.loc[d("2016-05-13") :]
    assert before[["eps_ttm", "fund_gross_margin", "fund_days_since_fs"]].isna().all().all()
    assert after["fund_gross_margin"].tolist() == [0.5, 0.5]
    assert after["fund_days_since_fs"].tolist() == [0.0, 1.0]
    assert after[["eps_ttm", "fund_gm_yoy_chg", "fund_rev_yoy"]].isna().all().all()
    assert after["fund_rev_mom"].tolist() == pytest.approx([0.1, 0.1])


def test_a_late_earlier_quarter_delays_only_the_signals_that_read_it() -> None:
    for period, announce in (
        ("2015-Q1", "2015-05-15"),
        ("2015-Q2", "2015-08-14"),
        ("2015-Q3", "2016-04-05"),
    ):
        statement(period, announce, 2.0)
    statement("2014-Q4", "2015-03-31")
    statement("2015-Q4", "2016-03-31", eps=5.0, gross_margin=0.5)

    attached = attach(panel_of("2330", "2016-03-30", "2016-04-06")).set_index("date")

    assert math.isnan(attached["fund_gross_margin"].iloc[0])
    assert attached["fund_gross_margin"].iloc[1:].tolist() == [0.5] * 5
    assert attached["eps_ttm"].iloc[:4].isna().all()
    assert attached["eps_ttm"].iloc[4:].tolist() == [11.0, 11.0]
    assert attached["fund_days_since_fs"].iloc[1:].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_an_older_month_announced_late_does_not_replace_a_newer_one() -> None:
    revenue("2330", "2015-01", [100.0 + k for k in range(16)])
    MonthlyRevenue.objects.filter(period="2015-01").update(announce_date=d("2016-05-16"))

    attached = attach(panel_of("2330", "2016-05-13", "2016-05-16"))

    assert attached["fund_rev_3m_yoy"].tolist() == pytest.approx(
        [(113 + 114 + 115) / (101 + 102 + 103) - 1] * 2
    )


def test_old_filings_go_stale_and_the_load_start_does_not_matter() -> None:
    revenue("2330", "2014-06", [100.0] * 12 + [110.0])
    statement("2015-Q1", "2015-05-15")

    edge = attach(panel_of("2330", "2015-09-10", "2015-09-11")).set_index("date")
    margin = attach(panel_of("2330", "2015-12-01", "2015-12-02")).set_index("date")
    full = attach(panel_of("2330", "2016-11-28", "2016-12-01"), "2015-01-02")
    late = attach(panel_of("2330", "2016-11-28", "2016-12-01"))

    assert edge["fund_rev_yoy"].tolist()[0] == pytest.approx(0.1)
    assert math.isnan(edge["fund_rev_yoy"].iloc[1])
    assert margin["fund_gross_margin"].tolist()[0] == 0.4
    assert math.isnan(margin["fund_gross_margin"].iloc[1])
    assert margin["fund_days_since_fs"].notna().all()
    assert full["fund_days_since_fs"].notna().all()
    pd.testing.assert_frame_equal(full, late)


def test_a_weekend_announcement_counts_from_the_next_trading_day() -> None:
    year_of_statements()
    statement("2015-Q4", "2016-03-26")
    panel = panel_of("2330", "2016-03-25", "2016-03-29")

    attached = attach_fundamentals(panel, CALENDAR, d("2016-03-25"), d("2016-03-29"))

    assert attached["fund_days_since_fs"].tolist()[1:] == [1.0, 2.0]


def test_a_filing_on_the_first_day_of_a_new_listing_is_visible() -> None:
    statement("2015-Q4", "2016-03-01", gross_margin=0.5)

    attached = attach(panel_of("2330", "2016-03-01", "2016-03-02", segment=2, age=0))

    assert attached["fund_gross_margin"].tolist() == [0.5, 0.5]


def test_a_month_waiting_for_a_late_input_shows_as_missing_not_as_an_older_month() -> None:
    revenue("2330", "2014-01", [100.0 + k for k in range(28)])
    MonthlyRevenue.objects.filter(period="2015-03").update(announce_date=d("2016-05-16"))

    attached = attach(panel_of("2330", "2016-04-15", "2016-05-16"))
    three, mom = attached["fund_rev_3m_yoy"].tolist(), attached["fund_rev_mom"].tolist()

    assert math.isnan(three[0])
    assert mom[0] == pytest.approx(126 / 125 - 1)
    assert three[-1] == pytest.approx((127 + 126 + 125) / (115 + 114 + 113) - 1)


def test_values_on_a_day_use_only_rows_announced_by_then() -> None:
    revenue("2330", "2014-01", [100.0 + 3 * k for k in range(28)])
    MonthlyRevenue.objects.filter(period="2015-03").update(announce_date=d("2016-05-16"))
    statement("2014-Q4", "2015-03-31")
    year_of_statements(2.0)
    statement("2015-Q4", "2016-03-31", eps=5.0, fin_type="individual", gross_margin=0.3)
    statement("2015-Q4", "2016-04-20", eps=6.0, gross_margin=0.5)
    statement("2014-Q3", "2016-05-03", eps=0.5)
    days = CALENDAR.between(d("2016-03-28"), d("2016-05-20"))
    columns = [*REVENUE_NAMES, *STATEMENT_NAMES, "fund_days_since_fs"]

    full = attach(panel_of("2330", "2016-03-28", "2016-05-20")).set_index("date")[columns]

    for day in reversed(days):
        MonthlyRevenue.objects.filter(announce_date__gt=day).delete()
        FinancialStatement.objects.filter(announce_date__gt=day).delete()
        seen = attach(panel_of("2330", day.isoformat(), day.isoformat())).set_index("date")
        pd.testing.assert_frame_equal(seen[columns], full.loc[[day]])


def test_the_lookback_covers_every_input_of_a_fresh_value() -> None:
    for year, eps in ((2014, 1.0), (2015, 2.0)):
        for quarter, announce in ((1, "05-15"), (2, "08-14"), (3, "11-14")):
            statement(f"{year}-Q{quarter}", f"{year}-{announce}", eps)
        statement(f"{year}-Q4", f"{year + 1}-03-31", eps)
    revenue("2330", "2015-04", [100.0] * 12 + [110.0] * 3)
    panel = panel_of("2330", "2016-09-09", "2016-09-09")

    late, early = attach(panel), attach(panel, "2013-01-02")

    pd.testing.assert_frame_equal(late, early)
    assert late["fund_eps_yoy"].tolist() == [1.0]
    assert late["fund_rev_3m_yoy"].tolist() == pytest.approx([0.1])
