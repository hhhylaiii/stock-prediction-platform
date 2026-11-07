from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
from apps.market.models import StockMaster, TradingCalendar
from apps.research.features.fundamental import crossed_name
from apps.research.features.residual import observable_name
from apps.research.management.commands import residual_coverage
from tests.research.conftest import weekday_calendar

pytestmark = pytest.mark.django_db
FIRST, LAST = date(2016, 1, 1), date(2017, 12, 31)


def stocks() -> None:
    StockMaster.objects.create(stock_id="2330", name="2330", market="TWSE", list_date=FIRST)
    StockMaster.objects.create(
        stock_id="1101", name="1101", market="TWSE", list_date=FIRST, delist_date=date(2018, 3, 1)
    )
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=open_day)
        for day, open_day in weekday_calendar(FIRST, LAST).all_days()
    )


def synthetic(calendar: TradingDays, start: date, end: date) -> pd.DataFrame:
    """Two stocks a day; 1101 lacks fund_rev_mom in 2016-03 and the holding after 2016-02.

    In 2016-03 only 1101 has fund_rev_yoy, and it equals the fallback, so it counts as filled.
    """
    low, high = max(start, calendar.start), min(end, calendar.end)
    days = calendar.between(low, high) if low <= high else ()
    rows: list[tuple[str, date, bool, float | None, float | None, float | None]] = []
    for day in days:
        month = day.strftime("%Y-%m")
        rows.append(("2330", day, True, 0.1, 0.5, None))
        gone = month == "2016-03"
        held = 0.4 if month < "2016-03" else None
        rows.append(("1101", day, True, None if gone else 0.2, held, 0.3 if gone else None))
    columns = ["fund_rev_mom", "chip_fi_holding_pct", "fund_rev_yoy"]
    frame = pd.DataFrame(rows, columns=["stock_id", "date", "in_universe", *columns])
    copies = dict.fromkeys(["fund_rev_3m_yoy", "fund_rev_yoy_q"], frame["fund_rev_yoy"])
    return frame.assign(**copies).astype(dict.fromkeys([*columns, *copies], "float64"))


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[tuple[date, date, Any]]:
    stocks()
    monkeypatch.setattr(residual_coverage, "MASKED_MONTHS", {})
    calls: list[tuple[date, date, Any]] = []

    def build(calendar: TradingDays, start: date, end: date, masked: Any) -> pd.DataFrame:
        calls.append((start, end, masked))
        return synthetic(calendar, start, end)

    monkeypatch.setattr(residual_coverage, "build_feature_frame", build)
    monkeypatch.setattr(residual_coverage, "observable_frame", lambda *_: unexplained())
    return calls


def unexplained() -> pd.DataFrame:
    return pd.DataFrame(columns=["stock_id", "date", observable_name("fund_rev_yoy")])


def test_coverage_builds_each_year_unmasked_and_names_the_affected_months(
    built: list[tuple[date, date, Any]], tmp_path: Path
) -> None:
    output = tmp_path / "coverage.csv"
    out = StringIO()

    call_command(
        "residual_coverage",
        columns="fund_rev_mom,chip_fi_holding_pct",
        start="2016-01-01",
        end="2017-12-31",
        output=str(output),
        stdout=out,
    )

    assert built == [
        (date(2016, 1, 1), date(2016, 12, 31), {}),
        (date(2017, 1, 1), date(2017, 12, 31), {}),
    ]
    text = out.getvalue()
    assert "fund_rev_mom: 1 of 24 months affected (4%)" in text
    assert '    "fund_rev_mom": frozenset({"2016-03"}),' in text
    assert "chip_fi_holding_pct: 22 of 24 months affected (92%)" in text
    assert text.count("warning: over 20% of the months") == 1
    assert text.index("warning") > text.index("chip_fi_holding_pct: 22")
    table = pd.read_csv(output)
    assert list(table.columns) == [
        "column",
        "month",
        "current",
        "delisted",
        "present_current",
        "present_delisted",
        "filled_current",
        "filled_delisted",
        "rows_current",
        "rows_delisted",
    ]
    assert table["filled_current"].isna().all()
    assert len(table) == 48
    march = table[(table["column"] == "fund_rev_mom") & (table["month"] == "2016-03")].iloc[0]
    assert (march["current"], march["delisted"]) == (1.0, 0.0)
    assert (march["rows_current"], march["rows_delisted"]) == (23, 23)


def test_coverage_defaults_to_the_rule_columns_and_refuses_unknown_ones(
    built: list[tuple[date, date, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, masked = StringIO(), StringIO()

    call_command("residual_coverage", start="2016-03-01", end="2016-04-30", stdout=out)
    monkeypatch.setattr(residual_coverage, "MASKED_MONTHS", {"fund_rev_mom": frozenset()})
    call_command("residual_coverage", start="2016-03-01", end="2016-04-30", stdout=masked)

    assert "fund_rev_yoy: 1 of 2 months affected (50%)" in out.getvalue()
    assert (
        "fund_rev_yoy: filled from fund_rev_yoy_q in the months left unmasked, "
        "current 0.00%, delisted 0.00%"
    ) in out.getvalue()
    assert "fills still lean" not in out.getvalue()
    assert '    "fund_rev_3m_yoy": frozenset({"2016-03"}),' in out.getvalue()
    assert "chip_fi_holding_pct" not in out.getvalue()
    assert "fund_rev_mom" not in out.getvalue()
    assert "fund_rev_mom: 1 of 2 months affected (50%)" in masked.getvalue()
    with pytest.raises(CommandError, match="not feature columns"):
        call_command("residual_coverage", columns="fwd_rank_pct", stdout=StringIO())
    with pytest.raises(CommandError, match="after"):
        call_command("residual_coverage", start="2017-01-01", end="2016-01-01", stdout=StringIO())


def test_coverage_reads_2010_to_2020_by_default_and_names_an_empty_period(
    built: list[tuple[date, date, Any]],
) -> None:
    out, empty = StringIO(), StringIO()

    call_command("residual_coverage", stdout=out)
    call_command("residual_coverage", start="2018-01-01", end="2018-12-31", stdout=empty)

    assert [start for start, _, _ in built[:11]] == [date(year, 1, 1) for year in range(2010, 2021)]
    assert built[10][1] == date(2020, 12, 31)
    assert "fund_rev_yoy: 1 of 24 months affected (4%)" in out.getvalue()
    assert "fund_rev_yoy: no universe rows in the period" in empty.getvalue()


def test_coverage_counts_a_gap_with_a_known_cause_as_covered(
    built: list[tuple[date, date, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[tuple[date, date]] = []

    def explained(calendar: TradingDays, start: date, end: date) -> pd.DataFrame:
        asked.append((start, end))
        low, high = max(start, calendar.start), min(end, calendar.end)
        days = calendar.between(low, high) if low <= high else ()
        rows = [(stock, day, True, True) for day in days for stock in ("2330", "1101")]
        names = [observable_name("fund_rev_mom"), observable_name("fund_rev_yoy")]
        return pd.DataFrame(rows, columns=["stock_id", "date", *names])

    monkeypatch.setattr(residual_coverage, "observable_frame", explained)
    out = StringIO()

    call_command(
        "residual_coverage", columns="fund_rev_mom,fund_rev_yoy,fund_rev_3m_yoy", stdout=out
    )

    assert asked[:2] == [
        (date(2010, 1, 1), date(2010, 12, 31)),
        (date(2011, 1, 1), date(2011, 12, 31)),
    ]
    assert "fund_rev_mom: 0 of 24 months affected (0%)" in out.getvalue()
    assert "fund_rev_yoy: 1 of 24 months affected (4%)" in out.getvalue()
    assert "fund_rev_3m_yoy: 1 of 24 months affected (4%)" in out.getvalue()


def test_coverage_counts_gaps_read_across_report_bases_and_warns_when_they_lean_on_delisted(
    built: list[tuple[date, date, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def crossed(calendar: TradingDays, start: date, end: date) -> pd.DataFrame:
        low, high = max(start, calendar.start), min(end, calendar.end)
        days = calendar.between(low, high) if low <= high else ()
        rows = [(stock, day, stock == "1101") for day in days for stock in ("2330", "1101")]
        return pd.DataFrame(rows, columns=["stock_id", "date", crossed_name("fund_rev_yoy_q")])

    def build(calendar: TradingDays, start: date, end: date, masked: Any) -> pd.DataFrame:
        return synthetic(calendar, start, end).assign(fund_rev_yoy_q=float("nan"))

    monkeypatch.setattr(residual_coverage, "observable_frame", crossed)
    monkeypatch.setattr(residual_coverage, "build_feature_frame", build)
    out = StringIO()

    call_command("residual_coverage", columns="fund_rev_yoy_q,fund_rev_yoy", stdout=out)

    text = out.getvalue()
    assert "fund_rev_yoy_q: gaps read across report bases, current 0.00%, delisted 100.00%" in text
    assert "warning: these gaps lean on stocks later delisted; report before keeping them" in text
    assert "fund_rev_yoy: gaps read across" not in text


@pytest.mark.parametrize(
    ("current", "delisted", "warned"),
    [(20, 31, True), (30, 44, False), (4, 13, False), (4, 15, True), (0, 10, False)],
)
def test_the_basis_warning_needs_both_the_ratio_and_the_gap(
    current: int, delisted: int, warned: bool
) -> None:
    out = StringIO()
    counts = pd.Series(
        {"current": current, "delisted": delisted, "rows_current": 1000, "rows_delisted": 1000}
    )

    residual_coverage.Command(stdout=out)._basis("fund_cfo_to_ni", counts)

    assert ("warning: these gaps lean" in out.getvalue()) is warned


@pytest.mark.parametrize(
    ("current", "delisted", "warned"),
    [(0.01, 0.04, True), (0.01, 0.03, False), (0.0, 0.01, True), (0.0, 0.0, False)],
)
def test_the_fill_warning_reads_the_months_left_unmasked(
    current: float, delisted: float, warned: bool
) -> None:
    out = StringIO()
    coverage = pd.DataFrame(
        {
            "filled_current": [0.9, current],
            "filled_delisted": [0.9, delisted],
            "rows_current": [100, 100],
            "rows_delisted": [100, 100],
        },
        ["2016-03", "2016-04"],
    )

    residual_coverage.Command(stdout=out)._fills("fund_rev_yoy", coverage, ["2016-03"])

    assert f"current {current:.2%}, delisted {delisted:.2%}" in out.getvalue()
    assert ("warning: fills still lean on stocks later delisted" in out.getvalue()) is warned


def test_the_fill_line_says_so_when_every_month_is_affected() -> None:
    out = StringIO()
    coverage = pd.DataFrame(
        {
            "filled_current": [0.0],
            "filled_delisted": [0.9],
            "rows_current": [100],
            "rows_delisted": [100],
        },
        ["2016-03"],
    )

    residual_coverage.Command(stdout=out)._fills("fund_rev_yoy", coverage, ["2016-03"])

    expected = "fund_rev_yoy: every month is affected, so no fills are left unmasked\n"
    assert out.getvalue() == expected
