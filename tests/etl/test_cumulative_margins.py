import io
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command

from apps.etl.adapters.mops import STATEMENT_SCAN
from apps.etl.cumulative_margins import MARGIN_CHECK, clear_cumulative_margins
from apps.etl.models import DataQualityLog, EtlRun, RawMops
from apps.etl.mops_history import STATEMENT_HISTORY, STATEMENT_VIEWER
from apps.etl.statement_summary import REPORTS, SEASON_LABELS
from apps.market.models import FinancialStatement

H1_GROSS = 36_896 / 216_026
Q2_GROSS = (36_896 - 20_390) / (216_026 - 108_424)
H2_GROSS = (71_388 - 36_896) / (426_921 - 216_026)
Q2_OPERATING = (-57_182 - 11_132) / (216_026 - 108_424)


def viewer_page(fixture_path: Callable[[str], Path], season: int) -> str:
    path = fixture_path(f"mops/viewer/t164sb01_6211_2013_{season}_C.html")
    return path.read_bytes().decode("cp950")


def save_viewer(fixture_path: Callable[[str], Path], *seasons: int) -> None:
    for season in seasons:
        RawMops.objects.create(
            dataset=STATEMENT_VIEWER,
            request_params={
                "step": "1",
                "CO_ID": "6211",
                "SYEAR": "2013",
                "SSEASON": str(season),
                "REPORT_ID": "C",
            },
            payload={"html": viewer_page(fixture_path, season)},
            source_hash=f"6211{season}",
        )


def stored(period: str = "2013-Q2", fin_type: str = "consolidated", **values: Any) -> None:
    FinancialStatement.objects.create(
        stock_id="6211",
        period=period,
        fin_type=fin_type,
        announce_date=date(2014, 6, 3),
        **{"gross_margin": None, "op_margin": None, **values},
    )


def margins(period: str = "2013-Q2") -> tuple[float | None, float | None]:
    row = FinancialStatement.objects.get(stock_id="6211", period=period)
    return row.gross_margin, row.op_margin


def logged() -> dict[str, Any]:
    (row,) = DataQualityLog.objects.filter(check_name=MARGIN_CHECK)
    return dict(row.detail)


@pytest.mark.django_db
def test_a_year_to_date_margin_is_cleared_and_a_quarter_one_kept(
    fixture_path: Callable[[str], Path],
) -> None:
    save_viewer(fixture_path, 2)
    stored(gross_margin=H1_GROSS, op_margin=0.5)
    cleared = clear_cumulative_margins()
    assert margins() == (None, 0.5)
    assert cleared.kinds == {"cumulative": 1}
    assert logged()["changes"] == [
        {
            "period": "2013-Q2",
            "fin_type": "consolidated",
            "fields": {"gross_margin": [H1_GROSS, None, "cumulative"]},
        }
    ]
    assert logged()["unmatched"] == [
        {"period": "2013-Q2", "fin_type": "consolidated", "field": "op_margin", "value": 0.5}
    ]


@pytest.mark.django_db
def test_a_single_quarter_margin_is_kept(fixture_path: Callable[[str], Path]) -> None:
    save_viewer(fixture_path, 1, 2)
    stored(gross_margin=Q2_GROSS, op_margin=Q2_OPERATING)
    cleared = clear_cumulative_margins()
    gross, operating = margins()
    assert gross == pytest.approx(Q2_GROSS)
    assert operating == pytest.approx(Q2_OPERATING)
    assert (cleared.kinds, cleared.unmatched) == ({}, 0)
    assert not DataQualityLog.objects.filter(check_name=MARGIN_CHECK).exists()


@pytest.mark.django_db
def test_a_fourth_quarter_over_the_half_year_is_cleared(
    fixture_path: Callable[[str], Path],
) -> None:
    save_viewer(fixture_path, 2, 4)
    stored("2013-Q4", gross_margin=H2_GROSS)
    assert clear_cumulative_margins().kinds == {"2q": 1}
    assert margins("2013-Q4") == (None, None)


@pytest.mark.django_db
def test_other_report_kinds_first_quarters_and_rows_without_pages_are_not_judged(
    fixture_path: Callable[[str], Path],
) -> None:
    save_viewer(fixture_path, 1, 2)
    stored(fin_type="individual", gross_margin=H1_GROSS)
    stored("2013-Q1", gross_margin=20_390 / 108_424)
    stored("2013-Q3", gross_margin=0.5)
    cleared = clear_cumulative_margins()
    assert (cleared.kinds, cleared.unmatched, cleared.unchecked) == ({}, 0, 2)
    assert FinancialStatement.objects.filter(gross_margin__isnull=True).count() == 0


@pytest.mark.django_db
def test_a_rerun_keeps_the_values_cleared_before(fixture_path: Callable[[str], Path]) -> None:
    save_viewer(fixture_path, 2, 4)
    stored(gross_margin=H1_GROSS)
    clear_cumulative_margins()
    stored("2013-Q4", gross_margin=H2_GROSS)
    assert clear_cumulative_margins().kinds == {"2q": 1}
    changes = {entry["period"]: entry["fields"] for entry in logged()["changes"]}
    assert changes == {
        "2013-Q2": {"gross_margin": [H1_GROSS, None, "cumulative"]},
        "2013-Q4": {"gross_margin": [H2_GROSS, None, "2q"]},
    }


@pytest.mark.django_db
def test_quarters_the_daily_scan_read_are_left_alone(
    fixture_path: Callable[[str], Path],
) -> None:
    EtlRun.objects.create(
        source="mops",
        dataset=STATEMENT_SCAN,
        start_date=date(2013, 8, 1),
        end_date=date(2013, 8, 1),
    )
    save_viewer(fixture_path, 2)
    stored(gross_margin=H1_GROSS)
    assert clear_cumulative_margins().kinds == {}
    gross, operating = margins()
    assert gross == pytest.approx(H1_GROSS)
    assert operating is None


def second_quarter_summaries(fixture_path: Callable[[str], Path]) -> None:
    for report in REPORTS:
        for market in ("sii", "otc"):
            page = fixture_path(f"mops/history/{report}_{market}_102_01.html").read_text("utf-8")
            RawMops.objects.create(
                dataset=STATEMENT_HISTORY,
                request_params={"report": report, "TYPEK": market, "year": "102", "season": "02"},
                payload={"html": page.replace(SEASON_LABELS[1], SEASON_LABELS[2])},
                source_hash=f"{report}{market}",
            )


@pytest.mark.django_db
def test_the_command_judges_margins_against_the_summaries_too(
    fixture_path: Callable[[str], Path],
) -> None:
    second_quarter_summaries(fixture_path)
    FinancialStatement.objects.create(
        stock_id="4526",
        period="2013-Q2",
        fin_type="consolidated",
        announce_date=date(2014, 6, 3),
        gross_margin=320_083 / 1_452_830,
    )
    out = io.StringIO()
    call_command("overwrite_statements", "--dry-run", stdout=out)
    assert FinancialStatement.objects.get().gross_margin == pytest.approx(320_083 / 1_452_830)
    assert "1 margins would be cleared (cumulative 1)" in out.getvalue()
    call_command("overwrite_statements", stdout=out)
    assert FinancialStatement.objects.get().gross_margin is None


@pytest.mark.django_db
def test_only_the_stocks_asked_for_are_judged(fixture_path: Callable[[str], Path]) -> None:
    save_viewer(fixture_path, 2)
    second_quarter_summaries(fixture_path)
    stored(gross_margin=H1_GROSS)
    FinancialStatement.objects.create(
        stock_id="4526",
        period="2013-Q2",
        fin_type="consolidated",
        announce_date=date(2014, 6, 3),
        gross_margin=320_083 / 1_452_830,
    )
    assert clear_cumulative_margins(["4526"]).kinds == {"cumulative": 1}
    assert margins()[0] == pytest.approx(H1_GROSS)
