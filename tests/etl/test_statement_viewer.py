import re
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from apps.etl.errors import SchemaChangedError
from apps.etl.statement_summary import BALANCE, CASH_FLOW, INCOME, parse_summary, statement_rows
from apps.etl.statement_viewer import parse_viewer

REVENUE, PARENT, EPS = "營業收入", "淨利（淨損）歸屬於母公司業主", "基本每股盈餘（元）"
ASSETS, CFO = "資產總計", "營業活動之淨現金流入（流出）"
BALANCE_FIRST = (INCOME, BALANCE, CASH_FLOW)


@pytest.fixture
def page(fixture_path: Callable[[str], Path]) -> Callable[[str], str]:
    return lambda name: fixture_path(f"mops/viewer/{name}").read_bytes().decode("cp950")


def summary(page: Callable[[str], str], code: str, year: int, season: int) -> dict:
    found = parse_viewer(page(f"t164sb01_{code}_{year}_{season}_C.html"), code, f"{year}-Q{season}")
    assert found is not None
    return {report: {code: found[report]} for report in found}


def test_a_first_quarter_reads_its_only_columns(page: Callable[[str], str]) -> None:
    found = parse_viewer(page("t164sb01_6211_2013_1_C.html"), "6211", "2013-Q1")
    assert found is not None
    assert found[INCOME][REVENUE] == 108_424
    assert found[INCOME][EPS] == 1.16
    assert found[BALANCE]["負債總額"] == 356_043
    assert found[CASH_FLOW][CFO] == 44_669


def test_later_quarters_read_the_year_to_date_column(page: Callable[[str], str]) -> None:
    found = summary(page, "6211", 2013, 2)
    assert found[INCOME]["6211"][REVENUE] == 216_026
    assert found[INCOME]["6211"][PARENT] == -2_003
    assert found[BALANCE]["6211"]["資產總額"] == 924_439
    assert found[CASH_FLOW]["6211"][CFO] == 66_870


def test_the_annual_report_reads_the_whole_year(page: Callable[[str], str]) -> None:
    found = summary(page, "6211", 2013, 4)
    assert found[INCOME]["6211"][REVENUE] == 426_921
    assert found[BALANCE]["6211"]["資產總額"] == 965_381


def test_the_2019_layout_has_codes_english_titles_and_bracketed_losses(
    page: Callable[[str], str],
) -> None:
    found = summary(page, "1262", 2019, 2)
    assert found[INCOME]["1262"][REVENUE] == 1_827_258
    assert found[INCOME]["1262"][PARENT] == -250_072
    assert found[INCOME]["1262"][EPS] == -1.53
    assert found[BALANCE]["1262"][ASSETS] == 11_051_624
    assert found[BALANCE]["1262"]["歸屬於母公司業主之權益合計"] == 10_057_215
    assert found[CASH_FLOW]["1262"][CFO] == 109_522


def test_banks_and_brokers_name_revenue_and_profit_their_own_way(
    page: Callable[[str], str],
) -> None:
    bank = summary(page, "2847", 2015, 3)[INCOME]["2847"]
    assert (bank["淨收益"], bank["淨利（損）歸屬於母公司業主"], bank[EPS]) == (
        6_811_858,
        1_238_650,
        0.45,
    )
    broker = summary(page, "6022", 2016, 4)[INCOME]["6022"]
    assert broker[PARENT] == 949
    assert "收入" not in broker
    empty: dict[str, dict[str, dict[str, float | None]]] = {report: {} for report in BALANCE_FIRST}
    (bank_row,) = statement_rows(summary(page, "2847", 2015, 3), empty, 1)
    assert bank_row["op_margin"] is None


def test_the_viewer_gives_the_figures_the_summaries_give(
    page: Callable[[str], str], fixture_path: Callable[[str], Path]
) -> None:
    current = summary(page, "4526", 2013, 1)
    empty: dict[str, dict[str, dict[str, float | None]]] = {report: {} for report in current}
    (row,) = statement_rows(current, empty, 1)
    summaries = {
        report: parse_summary(
            fixture_path(f"mops/history/{report}_sii_102_01.html").read_text("utf-8"),
            "sii",
            1,
            report,
        )
        for report in current
    }
    published = {report: {"4526": summaries[report]["4526"]} for report in current}
    assert row == statement_rows(published, empty, 1)[0]
    assert row["gross_margin"] is not None
    assert row["op_margin"] is not None
    assert row["fin_type"] == "consolidated"
    assert (row["eps"], row["revenue"], row["net_income"]) == (
        Decimal("0.1200"),
        1_452_830_000,
        26_804_000,
    )
    assert row["operating_cf"] == -92_392_000
    assert row["debt_ratio"] == pytest.approx(0.51605126542187)


def test_a_second_quarter_against_the_first_is_the_three_month_column(
    page: Callable[[str], str],
) -> None:
    (row,) = statement_rows(summary(page, "6211", 2013, 2), summary(page, "6211", 2013, 1), 2)
    assert (row["revenue"], row["net_income"]) == (107_602_000, -50_158_000)
    assert row["eps"] == Decimal("-1.2100")


def test_a_quarter_never_filed_is_none(page: Callable[[str], str]) -> None:
    assert parse_viewer(page("t164sb01_6211_2019_1_C.html"), "6211", "2019-Q1") is None


def test_a_kind_of_report_never_filed_can_answer_with_the_query_form(
    page: Callable[[str], str],
) -> None:
    assert parse_viewer(page("t164sb01_1240_2015_2_A.html"), "1240", "2015-Q2") is None


def test_an_individual_report_on_the_full_page_is_an_individual_row(
    page: Callable[[str], str],
) -> None:
    found = parse_viewer(page("t164sb01_1240_2019_2_A.html"), "1240", "2019-Q2")
    assert found is not None
    current = {report: {"1240": found[report]} for report in found}
    empty: dict[str, dict[str, dict[str, float | None]]] = {report: {} for report in found}
    (row,) = statement_rows(current, empty, 2)
    assert row["fin_type"] == "individual"
    assert row["debt_ratio"] is not None
    assert (found[INCOME][REVENUE], found[INCOME]["本期淨利（淨損）"]) == (857_797, 74_510)
    assert found[INCOME]["營業毛利（毛損）"] == 145_617


def test_a_first_quarter_flow_may_be_headed_as_the_quarter(page: Callable[[str], str]) -> None:
    found = parse_viewer(page("t164sb01_2847_2015_1_C.html"), "2847", "2015-Q1")
    assert found is not None
    assert found[INCOME]["淨收益"] is not None
    assert found[CASH_FLOW][CFO] is not None


def test_a_later_quarter_headed_as_the_quarter_alone_is_refused(
    page: Callable[[str], str],
) -> None:
    text = (
        page("t164sb01_2847_2015_1_C.html")
        .replace("104年第1季", "104年第2季")
        .replace("2015年03月31日", "2015年06月30日")
        .replace("2015年第1季", "2015年第2季")
    )
    with pytest.raises(SchemaChangedError, match="綜合損益表 has no column for 2015-Q2"):
        parse_viewer(text, "2847", "2015-Q2")


def test_a_second_quarter_flow_may_be_headed_as_the_half_year(
    page: Callable[[str], str],
) -> None:
    found = parse_viewer(page("t164sb01_1469_2018_2_C.html"), "1469", "2018-Q2")
    assert found is not None
    assert found[INCOME][REVENUE] == 144_880
    assert found[CASH_FLOW][CFO] is not None


def test_a_third_quarter_headed_as_the_half_year_is_refused(page: Callable[[str], str]) -> None:
    text = (
        page("t164sb01_1469_2018_2_C.html")
        .replace("107年第2季", "107年第3季")
        .replace("2018年06月30日", "2018年09月30日")
    )
    with pytest.raises(SchemaChangedError, match="綜合損益表 has no column for 2018-Q3"):
        parse_viewer(text, "1469", "2018-Q3")


@pytest.mark.parametrize(
    ("fixture", "code", "period", "heading", "wrong"),
    [
        ("t164sb01_2847_2015_1_C.html", "2847", "2015-Q1", "2015年第1季", "2014年第1季"),
        ("t164sb01_1469_2018_2_C.html", "1469", "2018-Q2", "2018年上半年度", "2017年上半年度"),
        ("t164sb01_1469_2018_2_C.html", "1469", "2018-Q2", "2018年上半年度", "2018年第1季"),
    ],
)
def test_a_named_heading_of_another_year_or_quarter_is_refused(
    page: Callable[[str], str], fixture: str, code: str, period: str, heading: str, wrong: str
) -> None:
    text = page(fixture).replace(heading, wrong)
    with pytest.raises(SchemaChangedError, match=f"綜合損益表 has no column for {period}"):
        parse_viewer(text, code, period)


def test_another_company_or_quarter_is_refused(page: Callable[[str], str]) -> None:
    with pytest.raises(SchemaChangedError, match="is not 6212"):
        parse_viewer(page("t164sb01_6211_2013_1_C.html"), "6212", "2013-Q1")
    with pytest.raises(SchemaChangedError, match="2013-Q3"):
        parse_viewer(page("t164sb01_6211_2013_2_C.html"), "6211", "2013-Q3")


def test_a_statement_without_its_figures_is_refused(page: Callable[[str], str]) -> None:
    text = page("t164sb01_6211_2013_1_C.html").replace("營業活動之淨現金流入（流出）", "營業活動")
    with pytest.raises(SchemaChangedError, match="lacks 營業活動之淨現金流入"):
        parse_viewer(text, "6211", "2013-Q1")


def test_a_page_in_other_units_is_refused(page: Callable[[str], str]) -> None:
    text = page("t164sb01_6211_2013_1_C.html").replace("新臺幣仟元", "新臺幣元")
    with pytest.raises(SchemaChangedError, match="thousands"):
        parse_viewer(text, "6211", "2013-Q1")


def test_the_first_filled_row_of_a_title_counts(page: Callable[[str], str]) -> None:
    text = page("t164sb01_6211_2013_1_C.html")
    row = re.search(r"<tr[^>]*><td[^>]*>[\s\u3000]*營業收入合計</td>(?:<td[^>]*>[^<]*</td>)+", text)
    assert row is not None
    text = text.replace(row[0], row[0] + row[0].replace("108,424", "1"), 1)
    found = parse_viewer(text, "6211", "2013-Q1")
    assert found is not None
    assert found[INCOME][REVENUE] == 108_424


def test_flows_of_another_year_are_not_taken_for_the_quarter(page: Callable[[str], str]) -> None:
    text = page("t164sb01_6211_2013_2_C.html").replace(
        "2013年01月01日至2013年06月30日", "2011年01月01日至2011年06月30日"
    )
    with pytest.raises(SchemaChangedError, match="no column for 2013-Q2"):
        parse_viewer(text, "6211", "2013-Q2")


def test_a_statement_without_its_header_row_is_refused(page: Callable[[str], str]) -> None:
    text = page("t164sb01_6211_2013_1_C.html").replace("會計項目", "項目", 1)
    with pytest.raises(SchemaChangedError, match="資產負債表 has no header row"):
        parse_viewer(text, "6211", "2013-Q1")


def test_the_report_title_must_name_the_quarter_asked(page: Callable[[str], str]) -> None:
    text = page("t164sb01_6211_2013_1_C.html").replace("102年第1季", "102年第2季")
    with pytest.raises(SchemaChangedError, match=r"\('6211', '2013-Q2'\), is not 6211 2013-Q1"):
        parse_viewer(text, "6211", "2013-Q1")
