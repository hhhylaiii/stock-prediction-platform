from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from apps.etl.errors import SchemaChangedError
from apps.etl.statement_summary import (
    BALANCE,
    CASH_FLOW,
    INCOME,
    REPORTS,
    parse_summary,
    statement_rows,
)

FINMIND_Q2 = {
    "2330": (
        "consolidated",
        "27.2500",
        1270380250000,
        0.67721,
        0.60344,
        706561938000,
        0.11467,
        0.30944,
        783364977000,
    ),
    "2882": (
        "consolidated",
        "2.8000",
        88335024000,
        None,
        None,
        44676497000,
        0.04618,
        0.92448,
        -92246925000,
    ),
    "6005": ("consolidated", "2.0500", None, None, None, 5040253000, 0.10077, 0.89204, -1274177000),
    "2816": ("individual", "6.0600", None, None, None, 1355743000, 0.14964, 0.64619, 428516000),
    "2801": ("consolidated", "0.5100", None, None, None, 6093217000, 0.02674, 0.93503, 8014811000),
    "2820": ("individual", "0.5900", None, None, None, 796302000, 0.02857, 0.89791, -5219013000),
    "6488": (
        "consolidated",
        "7.9000",
        15214310000,
        0.20641,
        0.09352,
        3779084000,
        0.03976,
        0.55880,
        2138764000,
    ),
    "8069": (
        "consolidated",
        "3.2300",
        10216093000,
        0.58725,
        0.34667,
        3733035000,
        0.05326,
        0.36317,
        -25409000,
    ),
}
FIELDS = (
    "fin_type",
    "eps",
    "revenue",
    "gross_margin",
    "op_margin",
    "net_income",
    "roe",
    "debt_ratio",
    "operating_cf",
)


@pytest.fixture
def page(fixture_path: Callable[[str], Path]) -> Callable[[str, str, int], str]:
    def _read(report: str, market: str, season: int) -> str:
        name = f"mops/summary/{report}_{market}_115_{season:02d}.html"
        return fixture_path(name).read_text(encoding="utf-8")

    return _read


def summaries(page: Callable[[str, str, int], str], season: int) -> dict[str, dict[str, Any]]:
    return {
        report: {
            code: values
            for market in ("sii", "otc")
            for code, values in parse_summary(page(report, market, season), market, season).items()
        }
        for report in REPORTS
    }


def by_code(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["stock_id"]: row for row in rows}


def test_a_summary_gives_each_companys_year_to_date_figures(
    page: Callable[[str, str, int], str],
) -> None:
    income = parse_summary(page(INCOME, "sii", 2), "sii", 2)
    assert income["2330"]["營業收入"] == 2_404_483_690
    assert income["2330"]["基本每股盈餘（元）"] == 49.33
    assert income["2816"]["淨利（損）歸屬於母公司業主"] is None
    assert income["1409"]["收入"] == 22_040_585
    balance = parse_summary(page(BALANCE, "sii", 2), "sii", 2)
    assert balance["2801"]["資產總額"] == 3_535_310_278
    assert balance["2330"]["資產總計"] == 9_375_654_727


def test_no_filings_yet_is_an_empty_answer(page: Callable[[str, str, int], str]) -> None:
    assert parse_summary(page(INCOME, "sii", 4), "sii", 4) == {}


@pytest.mark.parametrize(("market", "season"), [("otc", 2), ("sii", 1)])
def test_a_page_of_another_market_or_season_fails(
    page: Callable[[str, str, int], str], market: str, season: int
) -> None:
    with pytest.raises(SchemaChangedError, match="is not"):
        parse_summary(page(INCOME, "sii", 2), market, season)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (">2,404,483,690<", ">2,404,483,69O<", "non-numeric"),
        (">台積電</td>", ">台積電</td><td>1</td>", "cells"),
        ("單位：新台幣仟元", "單位：新台幣元", "thousands"),
    ],
)
def test_a_changed_layout_fails(
    page: Callable[[str, str, int], str], old: str, new: str, message: str
) -> None:
    text = page(INCOME, "sii", 2)
    assert old in text
    with pytest.raises(SchemaChangedError, match=message):
        parse_summary(text.replace(old, new, 1), "sii", 2)


def test_a_company_listed_twice_fails(page: Callable[[str, str, int], str]) -> None:
    text = page(INCOME, "sii", 2)
    row = text[text.index("<tr class='", text.index(">2330<") - 200) :]
    row = row[: row.index("</tr>") + 5]
    with pytest.raises(SchemaChangedError, match="twice"):
        parse_summary(text.replace(row, row + row, 1), "sii", 2)


def test_a_page_without_any_report_fails() -> None:
    with pytest.raises(SchemaChangedError, match="no title"):
        parse_summary("<html><body>FOR SECURITY REASONS</body></html>", "sii", 2)


def test_second_quarter_figures_match_the_finmind_backfill(
    page: Callable[[str, str, int], str],
) -> None:
    rows = by_code(statement_rows(summaries(page, 2), summaries(page, 1), 2))
    for code, expected in FINMIND_Q2.items():
        row = rows[code]
        got = tuple(row[field] for field in FIELDS)
        assert got[0] == expected[0], code
        assert abs(got[1] - Decimal(expected[1])) <= Decimal("0.02"), code
        assert got[2] == expected[2], code
        for index in (3, 4, 6, 7):
            assert got[index] == (
                None if expected[index] is None else pytest.approx(expected[index], abs=1e-5)
            ), (code, FIELDS[index])
        assert (got[5], got[8]) == (expected[5], expected[8]), code


def test_other_industries_report_revenue_as_income(page: Callable[[str, str, int], str]) -> None:
    row = by_code(statement_rows(summaries(page, 2), summaries(page, 1), 2))["1409"]
    assert (row["revenue"], row["gross_margin"], row["op_margin"]) == (12_027_340_000, None, None)
    assert row["debt_ratio"] == pytest.approx(0.766052, abs=1e-6)


def test_a_switch_of_report_type_leaves_the_quarters_flows_empty(
    page: Callable[[str, str, int], str],
) -> None:
    row = by_code(statement_rows(summaries(page, 2), summaries(page, 1), 2))["5547"]
    assert row["fin_type"] == "consolidated"
    flows = ("eps", "revenue", "gross_margin", "op_margin", "net_income", "roe", "operating_cf")
    assert all(row[field] is None for field in flows)
    assert row["debt_ratio"] is not None


def test_quarterly_eps_is_the_difference_while_shares_hold_steady(
    page: Callable[[str, str, int], str],
) -> None:
    current, previous = summaries(page, 2), summaries(page, 1)
    row = by_code(statement_rows(current, previous, 2))["2882"]
    eps_now = current[INCOME]["2882"]["基本每股盈餘（元）"]
    eps_before = previous[INCOME]["2882"]["基本每股盈餘（元）"]
    assert row["eps"] == Decimal(str(round(eps_now - eps_before, 2)))


def test_quarterly_eps_follows_net_income_when_shares_change(
    page: Callable[[str, str, int], str],
) -> None:
    current, previous = summaries(page, 2), summaries(page, 1)
    row = by_code(statement_rows(current, previous, 2))["5386"]
    income = current[INCOME]["5386"]
    net_now = income["本期淨利（淨損）"]
    net_q2 = net_now - previous[INCOME]["5386"]["本期淨利（淨損）"]
    assert row["eps"] == Decimal(str(round(net_q2 * income["基本每股盈餘（元）"] / net_now, 2)))
    assert abs(row["eps"] - Decimal("14.22")) < Decimal("1.3")


def test_a_company_not_yet_in_every_summary_waits_for_the_next_scan(
    page: Callable[[str, str, int], str],
) -> None:
    current, previous = summaries(page, 2), summaries(page, 1)
    partial = {**current, BALANCE: {k: v for k, v in current[BALANCE].items() if k != "2330"}}
    assert "2330" not in by_code(statement_rows(partial, previous, 2))
    partial = {**current, CASH_FLOW: {k: v for k, v in current[CASH_FLOW].items() if k != "2882"}}
    assert "2882" not in by_code(statement_rows(partial, previous, 2))


def synthetic(net: float, eps: float) -> dict[str, dict[str, dict[str, float | None]]]:
    figures: dict[str, float | None] = {
        "本期淨利（淨損）": net,
        "基本每股盈餘（元）": eps,
        "營業收入": 1000.0,
    }
    return {INCOME: {"9999": figures}, BALANCE: {"9999": {}}, CASH_FLOW: {"9999": {}}}


@pytest.mark.parametrize(
    ("now", "before", "eps"),
    [
        ((3.0, 0.0), (1.0, 0.01), Decimal("-0.0100")),
        ((3.0, 0.03), (1.0, 0.0), Decimal("0.0300")),
        ((0.0, 0.0), (1.0, 0.01), Decimal("-0.0100")),
        ((3.0, 0.03), (0.0, 0.0), Decimal("0.0300")),
    ],
)
def test_a_year_to_date_eps_or_profit_of_zero_takes_the_eps_difference(
    now: tuple[float, float], before: tuple[float, float], eps: Decimal
) -> None:
    rows = statement_rows(synthetic(*now), synthetic(*before), 2)
    assert rows[0]["eps"] == eps


def test_the_first_quarter_keeps_its_year_to_date_figures(
    page: Callable[[str, str, int], str],
) -> None:
    current = summaries(page, 1)
    empty: dict[str, dict[str, Any]] = {report: {} for report in REPORTS}
    row = by_code(statement_rows(current, empty, 1))["2330"]
    assert row["revenue"] == round(current[INCOME]["2330"]["營業收入"] * 1000)
    assert row["eps"] == Decimal("22.0800")
    assert row["operating_cf"] == round(
        current[CASH_FLOW]["2330"]["營業活動之淨現金流入（流出）"] * 1000
    )
    assert row["roe"] is None
