import io
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command

from apps.etl.adapters.mops import STATEMENT_SCAN
from apps.etl.cumulative_margins import MARGIN_CHECK, _judge, clear_cumulative_margins, spans
from apps.etl.models import DataQualityLog, EtlRun, RawMops
from apps.etl.mops_history import STATEMENT_HISTORY, STATEMENT_VIEWER
from apps.etl.statement_summary import GROSS_PROFIT, REPORTS, REVENUE, SEASON_LABELS
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
    assert logged()["unmatched"] == []
    assert logged()["untried"] == [
        {"period": "2013-Q2", "fin_type": "consolidated", "field": "op_margin", "value": 0.5}
    ]
    assert DataQualityLog.objects.get(check_name=MARGIN_CHECK).date == date(2013, 6, 30)


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
    stored(fin_type="individual", revenue=1, gross_margin=H1_GROSS)
    stored("2013-Q1", gross_margin=20_390 / 108_424)
    stored("2013-Q3", revenue=1, gross_margin=0.5)
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
    stored("2013-Q3", revenue=1, gross_margin=0.5)
    cleared = clear_cumulative_margins(["4526"])
    assert (cleared.kinds, cleared.unchecked) == ({"cumulative": 1}, 0)
    assert margins()[0] == pytest.approx(H1_GROSS)


@pytest.mark.django_db
def test_before_ifrs_a_margin_beside_a_left_out_revenue_is_cleared_unverified() -> None:
    stored("2012-Q2", gross_margin=0.3, op_margin=0.1)
    stored("2012-Q3", revenue=100, gross_margin=0.3)
    assert clear_cumulative_margins().kinds == {"unverified": 2}
    assert (margins("2012-Q2"), margins("2012-Q3")) == ((None, None), (0.3, None))
    assert logged()["changes"][0]["fields"]["gross_margin"] == [0.3, None, "unverified"]


@pytest.mark.django_db
def test_a_rerun_drops_notes_the_new_pages_settle(fixture_path: Callable[[str], Path]) -> None:
    save_viewer(fixture_path, 2)
    stored(op_margin=Q2_OPERATING)
    assert clear_cumulative_margins().untried == 1
    save_viewer(fixture_path, 1)
    assert clear_cumulative_margins().untried == 0
    assert margins()[1] == pytest.approx(Q2_OPERATING)
    assert not DataQualityLog.objects.filter(check_name=MARGIN_CHECK).exists()


@pytest.mark.django_db
@pytest.mark.parametrize(("offset", "kinds"), [(5e-5, {"cumulative": 1}), (2e-4, {})])
def test_a_margin_matches_within_one_hundredth_of_a_point(
    fixture_path: Callable[[str], Path], offset: float, kinds: dict[str, int]
) -> None:
    save_viewer(fixture_path, 2)
    stored(gross_margin=H1_GROSS + offset)
    assert clear_cumulative_margins().kinds == kinds


def test_a_span_without_revenue_is_skipped_and_a_tie_keeps_the_quarter() -> None:
    revenue, gross = REVENUE[0], GROSS_PROFIT[0]
    empty_first = {1: {revenue: 0.0, gross: 0.0}, 2: {revenue: 100.0, gross: 30.0}}
    assert spans(0.3, 2, empty_first, GROSS_PROFIT) == [1, 2]
    flat = {1: {revenue: 100.0, gross: 30.0}, 2: {revenue: 100.0, gross: 30.0}}
    assert spans(0.3, 2, flat, GROSS_PROFIT) == [2]


@pytest.mark.django_db
def test_a_margin_tried_against_its_own_quarter_and_matching_nothing_is_unmatched(
    fixture_path: Callable[[str], Path],
) -> None:
    save_viewer(fixture_path, 1, 2)
    stored(gross_margin=0.5)
    assert clear_cumulative_margins().unmatched == 1
    assert logged()["unmatched"] == [
        {"period": "2013-Q2", "fin_type": "consolidated", "field": "gross_margin", "value": 0.5}
    ]
    assert logged()["untried"] == []


def test_a_margin_matching_its_own_quarter_and_more_is_kept() -> None:
    revenue, gross = REVENUE[0], GROSS_PROFIT[0]
    income = {1: {revenue: 0.0, gross: 0.0}, 2: {revenue: 100.0, gross: 30.0}}
    row = FinancialStatement(stock_id="6211", period="2013-Q2", gross_margin=0.3)
    assert _judge(row, income).fields == {}


@pytest.mark.django_db
def test_a_field_cleared_again_keeps_the_value_it_had_the_first_time(
    fixture_path: Callable[[str], Path],
) -> None:
    save_viewer(fixture_path, 2)
    stored(gross_margin=H1_GROSS)
    clear_cumulative_margins()
    FinancialStatement.objects.filter(period="2013-Q2").update(gross_margin=H1_GROSS)
    clear_cumulative_margins()
    (entry,) = logged()["changes"]
    assert entry["fields"] == {"gross_margin": [H1_GROSS, None, "cumulative"]}


@pytest.mark.django_db
def test_without_mops_figures_a_margin_beside_a_left_out_revenue_is_cleared_unjudged() -> None:
    stored("2013-Q3", gross_margin=0.3)
    stored("2013-Q4", revenue=100, gross_margin=0.3)
    cleared = clear_cumulative_margins()
    assert (cleared.kinds, cleared.unchecked) == ({"unjudged": 1}, 1)
    assert (margins("2013-Q3")[0], margins("2013-Q4")[0]) == (None, 0.3)


@pytest.mark.django_db
def test_a_stock_with_a_viewer_page_that_does_not_parse_is_not_judged(
    fixture_path: Callable[[str], Path],
) -> None:
    save_viewer(fixture_path, 2)
    RawMops.objects.filter(dataset=STATEMENT_VIEWER).update(payload={"html": "<html>blank</html>"})
    stored(gross_margin=H1_GROSS)
    cleared = clear_cumulative_margins()
    assert (cleared.kinds, cleared.unchecked, cleared.unreadable) == ({}, 1, ["6211"])
    assert margins()[0] == pytest.approx(H1_GROSS)
