from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.models import StockMaster
from apps.research.features.build import FEATURE_COLUMNS
from apps.research.phase3_report import TABLE_COLUMNS
from tests.research.test_screening import STOCKS, stored

pytestmark = pytest.mark.django_db
__all__ = ["stored"]


def before_table(path: Path) -> Path:
    table = pd.DataFrame(
        {
            "current": 0.0,
            "delisted": 0.5,
            "rows_current": 100,
            "rows_delisted": 10,
            "ratio": float("nan"),
            "gap_pp": 50.0,
            "flag": True,
        },
        index=pd.Index(["fund_ep"], name="feature"),
    )
    table.to_csv(path)
    return path


def test_the_report_covers_gaps_fallbacks_and_monthly_ic(stored: Path) -> None:
    StockMaster.objects.create(
        stock_id=STOCKS[0],
        name="x",
        market="TWSE",
        industry="水泥工業",
        delist_date=date(2018, 1, 2),
    )
    call_command("ic_report", run_key="screen1", output=str(stored / "ic.md"), stdout=StringIO())
    out = StringIO()

    call_command(
        "phase3_report",
        run_key="screen1",
        before=str(before_table(stored / "before.csv")),
        start="2016-01-01",
        end="2016-12-31",
        output=str(stored / "data.md"),
        appendix=str(stored / "appendix"),
        stdout=out,
    )

    text = (stored / "data.md").read_text()
    assert "screen1.screening" in text
    assert "| fund_ep | 50.0%／0.0% |" in text
    assert "### 5 日" in text
    assert "### 20 日" in text
    after = pd.read_csv(stored / "appendix" / "missing-after.csv", index_col=0)
    assert list(after.columns) == TABLE_COLUMNS
    assert list(after.index) == FEATURE_COLUMNS
    assert after["rows_delisted"].iloc[0] > 0
    assert after["rows_current"].iloc[0] > after["rows_delisted"].iloc[0]
    monthly = pd.read_csv(stored / "appendix" / "monthly-ic.csv")
    assert set(monthly["horizon"]) == {5, 20}
    fills = pd.read_csv(stored / "appendix" / "fills.csv", index_col=0)
    assert list(fills.index) == ["fund_rev_yoy", "fund_rev_3m_yoy"]
    assert "screen1.screening" in out.getvalue()


def test_the_report_needs_a_recorded_screening_and_a_complete_before_table(stored: Path) -> None:
    with pytest.raises(CommandError, match="run ic_report first"):
        call_command(
            "phase3_report", run_key="none", before=str(stored / "x.csv"), stdout=StringIO()
        )
    call_command("ic_report", run_key="screen1", output=str(stored / "ic.md"), stdout=StringIO())
    broken = stored / "broken.csv"
    pd.DataFrame({"current": [0.0]}, index=["fund_ep"]).to_csv(broken)

    with pytest.raises(CommandError, match="lacks delisted"):
        call_command("phase3_report", run_key="screen1", before=str(broken), stdout=StringIO())
