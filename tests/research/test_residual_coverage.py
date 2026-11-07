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
    """Two stocks a day; 1101 lacks fund_rev_mom in 2016-03 and the holding after 2016-02."""
    days = calendar.between(start, end)
    rows: list[tuple[str, date, bool, float | None, float | None]] = []
    for day in days:
        month = day.strftime("%Y-%m")
        rows.append(("2330", day, True, 0.1, 0.5))
        gone = month == "2016-03"
        rows.append(("1101", day, True, None if gone else 0.2, 0.4 if month < "2016-03" else None))
    return pd.DataFrame(
        rows, columns=["stock_id", "date", "in_universe", "fund_rev_mom", "chip_fi_holding_pct"]
    ).astype({"fund_rev_mom": "float64", "chip_fi_holding_pct": "float64"})


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[tuple[date, date, Any]]:
    stocks()
    calls: list[tuple[date, date, Any]] = []

    def build(calendar: TradingDays, start: date, end: date, masked: Any) -> pd.DataFrame:
        calls.append((start, end, masked))
        return synthetic(calendar, start, end)

    monkeypatch.setattr(residual_coverage, "build_feature_frame", build)
    return calls


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
    assert list(table.columns) == ["column", "month", "current", "delisted"]
    assert len(table) == 48
    march = table[(table["column"] == "fund_rev_mom") & (table["month"] == "2016-03")].iloc[0]
    assert (march["current"], march["delisted"]) == (1.0, 0.0)


def test_coverage_defaults_to_the_masked_columns_and_refuses_unknown_ones(
    built: list[tuple[date, date, Any]],
) -> None:
    out = StringIO()

    call_command("residual_coverage", start="2016-03-01", end="2016-04-30", stdout=out)

    assert "fund_rev_mom: 1 of 2 months affected (50%)" in out.getvalue()
    assert "chip_fi_holding_pct" not in out.getvalue()
    with pytest.raises(CommandError, match="not feature columns"):
        call_command("residual_coverage", columns="fwd_rank_pct", stdout=StringIO())
    with pytest.raises(CommandError, match="after"):
        call_command("residual_coverage", start="2017-01-01", end="2016-01-01", stdout=StringIO())
