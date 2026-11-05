from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from apps.etl.viewer_history import viewer_frame, viewer_periods

NOT_FILED = "檔案不存在!"
COLUMNS = ["stock_id", "period", "fin_type", "revenue", "net_income", "operating_cf", "eps"]
YEAR = ["2013-Q1", "2013-Q2", "2013-Q3", "2013-Q4"]


def test_quarters_stop_at_the_last_one_ended_while_listed() -> None:
    periods = viewer_periods(
        date(2012, 1, 1), date(2026, 12, 31), date(2026, 11, 6), 45, date(2016, 1, 4)
    )
    assert (periods[0], periods[-1], len(periods)) == ("2013-Q1", "2015-Q4", 12)


def test_a_listed_stock_stops_at_the_quarters_the_scan_still_watches() -> None:
    periods = viewer_periods(date(2026, 1, 1), date(2026, 12, 31), date(2026, 10, 16), 45, None)
    assert periods == ["2026-Q1", "2026-Q2"]


@pytest.fixture
def pages(fixture_path: Callable[[str], Path]) -> dict[str, str]:
    folder = "mops/viewer/t164sb01_6211_{}_{}_C.html"
    return {
        period: fixture_path(folder.format(period[:4], period[-1])).read_bytes().decode("cp950")
        for period in YEAR
    }


def served(pages: dict[str, str], answers: dict[tuple[str, str], str] | None = None) -> tuple:
    asked: list[tuple[str, str]] = []

    def page(period: str, kind: str) -> str:
        asked.append((period, kind))
        if answers is not None:
            return answers.get((period, kind), NOT_FILED)
        return pages[period] if kind == "C" else NOT_FILED

    return page, asked


def test_every_missing_quarter_is_its_year_to_date_less_the_one_before(
    pages: dict[str, str],
) -> None:
    page, asked = served(pages)
    frame = viewer_frame("6211", YEAR, set(), page, COLUMNS)
    rows = {row["period"]: row for row in frame.to_dict("records")}
    assert (rows["2013-Q2"]["revenue"], rows["2013-Q3"]["revenue"]) == (107_602_000, 96_580_000)
    q4 = rows["2013-Q4"]
    assert (q4["revenue"], q4["net_income"], q4["operating_cf"]) == (
        114_315_000,
        54_225_000,
        13_520_000,
    )
    assert {row["fin_type"] for row in rows.values()} == {"consolidated"}
    assert asked == [(period, "C") for period in YEAR]
    assert frame.attrs["absent"] == []


def test_a_stored_quarter_is_read_only_to_convert_the_next(pages: dict[str, str]) -> None:
    page, asked = served(pages)
    frame = viewer_frame("6211", YEAR, {"2013-Q1", "2013-Q2"}, page, COLUMNS)
    assert list(frame["period"]) == ["2013-Q3", "2013-Q4"]
    assert frame.iloc[0]["revenue"] == 96_580_000
    assert [period for period, _ in asked] == ["2013-Q2", "2013-Q3", "2013-Q4"]


def test_a_quarter_never_filed_is_noted_and_its_successor_has_no_flows(
    pages: dict[str, str],
) -> None:
    answers = {(p, "C"): pages[p] for p in ("2013-Q1", "2013-Q2", "2013-Q4")}
    page, asked = served(pages, answers)
    frame = viewer_frame("6211", YEAR, set(), page, COLUMNS)
    assert frame.attrs["absent"] == ["2013-Q3"]
    q4 = frame.set_index("period").loc["2013-Q4"]
    assert (q4["revenue"], q4["operating_cf"]) == (None, None)
    assert ("2013-Q3", "A") in asked


def test_an_individual_filer_is_asked_for_that_kind_first_afterwards(
    pages: dict[str, str],
) -> None:
    answers = {(p, "A"): pages[p] for p in YEAR}
    page, asked = served(pages, answers)
    viewer_frame("6211", YEAR, set(), page, COLUMNS)
    assert asked == [
        ("2013-Q1", "C"),
        ("2013-Q1", "A"),
        ("2013-Q2", "A"),
        ("2013-Q3", "A"),
        ("2013-Q4", "A"),
    ]
