import io
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from django.core.management import call_command

from apps.etl.adapters.mops import STATEMENT_SCAN
from apps.etl.errors import SchemaChangedError
from apps.etl.models import DataQualityLog, EtlRun, RawMops
from apps.etl.mops_history import STATEMENT_HISTORY, STATEMENT_VIEWER
from apps.etl.statement_overwrite import (
    OVERWRITE_CHECK,
    overwrite_statements,
    summary_frame,
    viewer_frame_from_raw,
)
from apps.etl.statement_scan import SCAN_COLUMNS
from apps.etl.statement_summary import REPORTS, SEASON_LABELS
from apps.market.models import FinancialStatement

CONSOLIDATED = "consolidated"
STORED = {
    "eps": None,
    "revenue": 100_000_000,
    "gross_margin": 0.30,
    "op_margin": 0.10,
    "net_income": 8_000_000,
    "roe": 0.02,
    "debt_ratio": 0.40,
    "operating_cf": 5_000_000,
}


def stored(
    code: str = "1101", period: str = "2013-Q2", announced: date = date(2014, 6, 3), **values: Any
) -> FinancialStatement:
    return FinancialStatement.objects.create(
        stock_id=code,
        period=period,
        fin_type=CONSOLIDATED,
        announce_date=announced,
        **{**STORED, **values},
    )


def frame(*records: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {**dict.fromkeys(SCAN_COLUMNS), "stock_id": "1101", "period": "2013-Q2", **record}
        for record in records
    ]
    return pd.DataFrame(rows, columns=SCAN_COLUMNS, dtype=object)


def mops(**values: Any) -> dict[str, Any]:
    return {"fin_type": CONSOLIDATED, **STORED, "eps": Decimal("0.5000"), **values}


def logged() -> dict[str, list[dict[str, Any]]]:
    rows = DataQualityLog.objects.filter(check_name=OVERWRITE_CHECK)
    return {row.detail["stock_id"]: row.detail["changes"] for row in rows}


@pytest.mark.django_db
def test_figures_past_the_tolerance_are_replaced_and_blanks_filled() -> None:
    stored()
    changed = overwrite_statements(
        frame(mops(revenue=100_000_500, operating_cf=4_000_000, net_income=None, roe=0.025))
    )
    row = FinancialStatement.objects.get()
    assert changed == 1
    assert (row.revenue, row.operating_cf, row.net_income, row.roe) == (
        100_000_000,
        4_000_000,
        8_000_000,
        0.025,
    )
    assert row.eps == Decimal("0.5000")
    assert logged() == {
        "1101": [
            {
                "period": "2013-Q2",
                "fin_type": CONSOLIDATED,
                "fields": {
                    "eps": [None, "0.5000"],
                    "operating_cf": [5_000_000, 4_000_000],
                    "roe": [0.02, 0.025],
                },
            }
        ]
    }


@pytest.mark.django_db
def test_a_stored_eps_and_both_margins_stay_as_finmind_has_them() -> None:
    stored(eps=Decimal("0.4000"))
    changed = overwrite_statements(
        frame(mops(eps=Decimal("0.9000"), gross_margin=0.5, op_margin=0.2))
    )
    row = FinancialStatement.objects.get()
    assert changed == 0
    assert (row.eps, row.gross_margin, row.op_margin) == (Decimal("0.4000"), 0.30, 0.10)
    assert logged() == {}


@pytest.mark.django_db
def test_a_blank_margin_is_not_filled() -> None:
    stored(eps=Decimal("0.4000"), gross_margin=None, op_margin=None)
    assert overwrite_statements(frame(mops(gross_margin=0.5, op_margin=0.2))) == 0
    row = FinancialStatement.objects.get()
    assert (row.gross_margin, row.op_margin) == (None, None)


@pytest.mark.django_db
def test_a_debt_ratio_and_blank_figures_are_overwritten_and_filled() -> None:
    stored(debt_ratio=0.40, net_income=None, revenue=None)
    overwrite_statements(frame(mops(debt_ratio=0.45, net_income=7_000_000, revenue=90_000_000)))
    row = FinancialStatement.objects.get()
    assert (row.debt_ratio, row.net_income, row.revenue) == (0.45, 7_000_000, 90_000_000)


@pytest.mark.django_db
def test_a_statement_listed_twice_counts_once() -> None:
    stored()
    assert overwrite_statements(frame(mops(roe=0.03), mops(roe=0.03, debt_ratio=0.5))) == 1
    row = FinancialStatement.objects.get()
    assert (row.roe, row.debt_ratio) == (0.03, 0.5)
    assert logged()["1101"][0]["fields"] == {
        "eps": [None, "0.5000"],
        "roe": [0.02, 0.03],
        "debt_ratio": [0.40, 0.5],
    }


@pytest.mark.django_db
def test_quarters_from_the_first_one_the_daily_scan_read_are_left_alone() -> None:
    EtlRun.objects.create(
        source="mops",
        dataset=STATEMENT_SCAN,
        start_date=date(2026, 11, 5),
        end_date=date(2026, 11, 5),
    )
    stored(period="2026-Q2", announced=date(2026, 8, 14))
    stored(period="2026-Q3", announced=date(2026, 11, 5))
    changed = overwrite_statements(
        frame({**mops(roe=0.03), "period": "2026-Q2"}, {**mops(roe=0.03), "period": "2026-Q3"})
    )
    assert changed == 1
    roe = dict(FinancialStatement.objects.values_list("period", "roe"))
    assert roe == {"2026-Q2": 0.03, "2026-Q3": 0.02}


@pytest.mark.django_db
def test_rows_of_another_type_or_not_stored_are_left_alone() -> None:
    stored()
    other = mops(fin_type="individual", operating_cf=1)
    missing = {**mops(), "stock_id": "2330"}
    assert overwrite_statements(frame(other, missing)) == 0
    assert FinancialStatement.objects.count() == 1
    assert FinancialStatement.objects.get().operating_cf == 5_000_000


@pytest.mark.django_db
def test_a_rerun_keeps_finmind_first_value_and_the_changes_it_did_not_redo() -> None:
    stored()
    stored(period="2013-Q3")
    overwrite_statements(
        frame(mops(operating_cf=4_000_000), {**mops(operating_cf=3_000_000), "period": "2013-Q3"})
    )
    assert overwrite_statements(frame(mops(operating_cf=4_500_000))) == 1
    assert overwrite_statements(frame()) == 0
    changes = {entry["period"]: entry["fields"]["operating_cf"] for entry in logged()["1101"]}
    assert changes == {"2013-Q2": [5_000_000, 4_500_000], "2013-Q3": [5_000_000, 3_000_000]}
    (row,) = DataQualityLog.objects.filter(check_name=OVERWRITE_CHECK)
    assert row.date == date(2013, 9, 30)


@pytest.mark.django_db
def test_only_the_first_scan_sets_where_the_scanned_quarters_begin() -> None:
    for day in (date(2026, 4, 10), date(2026, 11, 5)):
        EtlRun.objects.create(source="mops", dataset=STATEMENT_SCAN, start_date=day, end_date=day)
    stored(period="2025-Q3", announced=date(2025, 11, 14))
    stored(period="2025-Q4", announced=date(2026, 3, 31))
    changed = overwrite_statements(
        frame({**mops(roe=0.03), "period": "2025-Q3"}, {**mops(roe=0.03), "period": "2025-Q4"})
    )
    assert changed == 1
    roe = dict(FinancialStatement.objects.values_list("period", "roe"))
    assert roe == {"2025-Q3": 0.03, "2025-Q4": 0.02}


def history_page(fixture_path: Callable[[str], Path], report: str, market: str) -> str:
    return fixture_path(f"mops/history/{report}_{market}_102_01.html").read_text("utf-8")


def save_history(
    fixture_path: Callable[[str], Path], year: str = "102", season: int = 1, **pages: str
) -> None:
    for report in REPORTS:
        for market in ("sii", "otc"):
            page = history_page(fixture_path, report, market)
            RawMops.objects.create(
                dataset=STATEMENT_HISTORY,
                request_params={
                    "report": report,
                    "TYPEK": market,
                    "year": year,
                    "season": f"{season:02d}",
                },
                payload={
                    "html": pages.get(f"{report}_{market}")
                    or page.replace(SEASON_LABELS[1], SEASON_LABELS[season])
                },
                source_hash=f"{report}{market}{year}{season}",
            )


@pytest.mark.django_db
def test_the_summaries_are_read_again_from_raw(fixture_path: Callable[[str], Path]) -> None:
    save_history(fixture_path)
    found = summary_frame().set_index("stock_id")
    assert set(found["period"]) == {"2013-Q1"}
    assert found.loc["4526", "revenue"] == 1_452_830_000
    assert found.loc["4526", "operating_cf"] == -92_392_000
    assert found.loc["3105", "revenue"] == 2_848_050_000


@pytest.mark.django_db
def test_a_later_quarter_takes_its_year_to_date_less_the_one_before(
    fixture_path: Callable[[str], Path],
) -> None:
    save_history(fixture_path)
    save_history(fixture_path, season=2)
    rows = {(row["stock_id"], row["period"]): row for row in summary_frame().to_dict("records")}
    q2 = rows[("4526", "2013-Q2")]
    assert (q2["revenue"], q2["operating_cf"]) == (0, 0)
    assert q2["roe"] is not None


@pytest.mark.django_db
def test_a_kept_summary_without_companies_is_refused(fixture_path: Callable[[str], Path]) -> None:
    unfiled = fixture_path("mops/history/t163sb20_otc_102_01_unfiled.html").read_text("utf-8")
    save_history(fixture_path, t163sb20_otc=unfiled)
    with pytest.raises(SchemaChangedError, match="t163sb20 otc 2013-Q1 kept no companies"):
        summary_frame()


@pytest.mark.django_db
def test_a_first_quarter_needs_no_quarter_before(fixture_path: Callable[[str], Path]) -> None:
    save_history(fixture_path, year="103")
    found = summary_frame().set_index("stock_id")
    assert set(found["period"]) == {"2014-Q1"}
    assert found.loc["4526", "operating_cf"] == -92_392_000


def viewer_page(fixture_path: Callable[[str], Path], season: int) -> str:
    path = fixture_path(f"mops/viewer/t164sb01_6211_2013_{season}_C.html")
    return path.read_bytes().decode("cp950")


def save_viewer(page: str, code: str = "6211", season: int = 1) -> None:
    RawMops.objects.create(
        dataset=STATEMENT_VIEWER,
        request_params={
            "step": "1",
            "CO_ID": code,
            "SYEAR": "2013",
            "SSEASON": str(season),
            "REPORT_ID": "C",
        },
        payload={"html": page},
        source_hash=f"{code}{season}",
    )


@pytest.mark.django_db
def test_a_later_quarter_without_its_quarter_before_keeps_only_its_balances(
    fixture_path: Callable[[str], Path],
) -> None:
    save_viewer(viewer_page(fixture_path, 2), season=2)
    found = viewer_frame_from_raw().set_index("period")
    assert list(found.index) == ["2013-Q2"]
    assert found.loc["2013-Q2", "debt_ratio"] is not None
    flows = ["revenue", "net_income", "eps", "roe", "operating_cf", "gross_margin", "op_margin"]
    assert found.loc["2013-Q2", flows].isna().all()


@pytest.mark.django_db
def test_a_stock_whose_kept_page_does_not_parse_is_set_aside(
    fixture_path: Callable[[str], Path],
) -> None:
    save_viewer(viewer_page(fixture_path, 1))
    save_viewer(viewer_page(fixture_path, 1), code="1333")
    found = viewer_frame_from_raw()
    assert set(found["stock_id"]) == {"6211"}
    assert list(found.attrs["unreadable"]) == ["1333"]


@pytest.mark.django_db
def test_the_page_stored_last_is_the_one_read(fixture_path: Callable[[str], Path]) -> None:
    RawMops.objects.create(
        dataset=STATEMENT_VIEWER,
        request_params={
            "step": "1",
            "CO_ID": "6211",
            "SYEAR": "2013",
            "SSEASON": "1",
            "REPORT_ID": "C",
        },
        payload={"html": viewer_page(fixture_path, 2)},
        source_hash="old",
        fetched_at=datetime(2026, 11, 6, tzinfo=UTC),
    )
    save_viewer(viewer_page(fixture_path, 1))
    found = viewer_frame_from_raw()
    assert found.attrs["unreadable"] == []
    assert list(found["period"]) == ["2013-Q1"]


@pytest.mark.django_db
def test_the_viewer_pages_are_read_again_from_raw(fixture_path: Callable[[str], Path]) -> None:
    for season in (1, 2):
        save_viewer(viewer_page(fixture_path, season), season=season)
    found = viewer_frame_from_raw().set_index("period")
    assert list(found.index) == ["2013-Q1", "2013-Q2"]
    assert (found.loc["2013-Q2", "revenue"], found.loc["2013-Q2", "net_income"]) == (
        107_602_000,
        -50_158_000,
    )


def save_both(fixture_path: Callable[[str], Path]) -> None:
    save_history(fixture_path)
    for season in (1, 2):
        save_viewer(viewer_page(fixture_path, season), season=season)
    save_viewer(viewer_page(fixture_path, 1), code="1333")
    stored("4526", "2013-Q1", operating_cf=1)
    stored("6211", "2013-Q2", revenue=1)


@pytest.mark.django_db
def test_the_command_overwrites_from_both_kinds_of_raw(
    fixture_path: Callable[[str], Path],
) -> None:
    save_both(fixture_path)
    out = io.StringIO()
    call_command("overwrite_statements", stdout=out)
    figures = dict(FinancialStatement.objects.values_list("stock_id", "operating_cf"))
    assert figures["4526"] == -92_392_000
    assert FinancialStatement.objects.get(stock_id="6211").revenue == 107_602_000
    assert "2 statements changed" in out.getvalue()
    assert "set aside 1333: their kept viewer pages do not parse" in out.getvalue()


@pytest.mark.django_db
def test_the_command_reads_only_the_stocks_asked_for(
    fixture_path: Callable[[str], Path],
) -> None:
    save_both(fixture_path)
    out = io.StringIO()
    call_command("overwrite_statements", "--stock", "6211", stdout=out)
    assert FinancialStatement.objects.get(stock_id="4526").operating_cf == 1
    assert FinancialStatement.objects.get(stock_id="6211").revenue == 107_602_000
    assert "1 statements changed" in out.getvalue()
    assert "set aside" not in out.getvalue()


@pytest.mark.django_db
def test_a_dry_run_counts_without_writing(fixture_path: Callable[[str], Path]) -> None:
    save_history(fixture_path)
    stored("4526", "2013-Q1", operating_cf=1)
    out = io.StringIO()
    call_command("overwrite_statements", "--dry-run", stdout=out)
    assert FinancialStatement.objects.get().operating_cf == 1
    assert not DataQualityLog.objects.filter(check_name=OVERWRITE_CHECK).exists()
    assert "1 statements would change" in out.getvalue()
