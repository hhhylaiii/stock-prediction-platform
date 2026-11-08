from collections.abc import Callable
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.models import StockMaster
from apps.research.features.build import CHECKED, MODEL_COLUMNS, current_name
from apps.research.features.fundamental import sourced_name
from apps.research.features.residual import FALLBACKS, OBSERVABLE, observable_name
from apps.research.features.store import existing_partition, export_year
from apps.research.management.commands import phase3_report
from apps.research.models import FeatureDaily, ModelRun
from apps.research.phase3_report import AFTER_COLUMNS, TABLE_COLUMNS
from tests.research.test_screening import STOCKS, stored

pytestmark = pytest.mark.django_db
__all__ = ["stored"]
Explained = Callable[..., pd.DataFrame]


def from_table(known: bool, drift: str = "") -> Explained:
    """observable_frame stand-in from feature_daily; drift fills or empties today's values."""

    def frame(calendar: object, start: date, end: date, **_: object) -> pd.DataFrame:
        rows = pd.DataFrame(
            list(
                FeatureDaily.objects.filter(date__range=(start, end)).values(
                    "stock_id", "date", "in_universe", *CHECKED
                )
            )
        )
        moved = (~rows["in_universe"] & (drift == "outside")) | (drift == "all")
        now = rows[CHECKED].where(~moved, rows[CHECKED].fillna(1.0), axis=0)
        if drift == "emptied":
            now = pd.DataFrame(float("nan"), index=rows.index, columns=CHECKED)
        return rows[["stock_id", "date"]].assign(
            **{observable_name(name): known for name in OBSERVABLE},
            **{sourced_name(name): 1.0 for name in FALLBACKS},
            **{current_name(name): now[name] for name in CHECKED},
        )

    return frame


@pytest.fixture(autouse=True)
def unexplained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase3_report, "observable_frame", from_table(known=False))
    monkeypatch.setattr(phase3_report, "MASKED_MONTHS", {})


def report(stored: Path, **options: str) -> pd.DataFrame:
    StockMaster.objects.get_or_create(
        stock_id=STOCKS[0],
        defaults={"name": "x", "market": "TWSE", "delist_date": date(2018, 1, 2)},
    )
    gone = FeatureDaily.objects.filter(stock_id=STOCKS[0], date__month=3)
    gone.update(fund_ep=None, fund_rev_yoy=None)
    export_year(2016, stored / "features", rebuilt=(date(2016, 1, 1), date(2016, 12, 31)))
    call_command("ic_report", run_key="screen1", output=str(stored / "ic.md"), stdout=StringIO())
    call_command(
        "phase3_report",
        run_key="screen1",
        before=str(before_table(stored / "before.csv")),
        start="2016-01-01",
        end="2016-12-31",
        output=str(stored / "data.md"),
        appendix=str(stored / "appendix"),
        stdout=StringIO(),
        **options,
    )
    return pd.read_csv(stored / "appendix" / "missing-after.csv", index_col=0)


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
    assert "`before.csv`" in text
    assert "--start/--end differ" in out.getvalue()
    assert "| fund_ep | 50.0%／0.0% |" in text
    assert "### 5 日" in text
    assert "### 20 日" in text
    assert "目前沒有遮罩欄位。" in text
    assert "| 特徵 | 遮罩月份 |" not in text
    after = pd.read_csv(stored / "appendix" / "missing-after.csv", index_col=0)
    assert list(after.columns) == AFTER_COLUMNS
    assert list(after.index) == MODEL_COLUMNS
    assert after["rows_delisted"].iloc[0] > 0
    assert after["rows_current"].iloc[0] > after["rows_delisted"].iloc[0]
    before = pd.read_csv(stored / "appendix" / "missing-before.csv", index_col=0)
    assert list(before.columns) == TABLE_COLUMNS
    assert list(before.index) == ["fund_ep"]
    monthly = pd.read_csv(stored / "appendix" / "monthly-ic.csv")
    assert set(monthly["horizon"]) == {5, 20}
    kept = ModelRun.objects.get(run_id="screen1.screening").feature_list
    header = next(line for line in text.splitlines() if line.startswith("| 月份 |"))
    assert set(header.strip("| ").split(" | ")[1:]) <= set(kept)
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


def test_the_report_refuses_features_rebuilt_after_the_screening(stored: Path) -> None:
    call_command("ic_report", run_key="screen1", output=str(stored / "ic.md"), stdout=StringIO())
    partition = existing_partition(stored / "features", 2016)
    partition.write_bytes(partition.read_bytes() + b"\0")

    with pytest.raises(CommandError, match="features changed since the screening"):
        call_command(
            "phase3_report",
            run_key="screen1",
            before=str(before_table(stored / "before.csv")),
            stdout=StringIO(),
        )


def test_the_report_adds_up_years_and_lists_every_screened_feature(stored: Path) -> None:
    january = list(FeatureDaily.objects.filter(date__month=1).values())
    FeatureDaily.objects.bulk_create(
        FeatureDaily(**(row | {"date": row["date"].replace(year=2015)})) for row in january
    )
    export_year(2015, stored / "features")
    call_command("ic_report", run_key="screen1", output=str(stored / "ic.md"), stdout=StringIO())

    call_command(
        "phase3_report",
        run_key="screen1",
        before=str(before_table(stored / "before.csv")),
        start="2015-01-01",
        end="2016-12-31",
        output=str(stored / "data.md"),
        appendix=str(stored / "appendix"),
        stdout=StringIO(),
    )

    after = pd.read_csv(stored / "appendix" / "missing-after.csv", index_col=0)
    universe = FeatureDaily.objects.filter(in_universe=True).count()
    assert (after["rows_current"] + after["rows_delisted"]).eq(universe).all()
    monthly = pd.read_csv(stored / "appendix" / "monthly-ic.csv")
    screened = ModelRun.objects.get(run_id="screen1.screening").params["ic"]
    assert set(monthly["feature"]) == set(screened)


def test_the_report_refuses_partitions_exported_without_the_source_gap_rules(
    stored: Path,
) -> None:
    partition = existing_partition(stored / "features", 2016)
    pd.read_parquet(partition).to_parquet(partition, index=False)
    call_command("ic_report", run_key="screen1", output=str(stored / "ic.md"), stdout=StringIO())

    with pytest.raises(CommandError, match="2016 partition was exported under other source-gap"):
        call_command(
            "phase3_report",
            run_key="screen1",
            before=str(before_table(stored / "before.csv")),
            start="2016-01-01",
            end="2016-12-31",
            output=str(stored / "data.md"),
            appendix=str(stored / "appendix"),
            stdout=StringIO(),
        )
    assert not (stored / "data.md").exists()


def test_the_report_refuses_partitions_stamped_with_other_rules(
    stored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_command("ic_report", run_key="screen1", output=str(stored / "ic.md"), stdout=StringIO())
    monkeypatch.setattr(phase3_report, "rules_signature", lambda: "{}")

    with pytest.raises(CommandError, match="rebuild features"):
        call_command(
            "phase3_report",
            run_key="screen1",
            before=str(before_table(stored / "before.csv")),
            start="2016-01-01",
            end="2016-12-31",
            output=str(stored / "data.md"),
            appendix=str(stored / "appendix"),
            stdout=StringIO(),
        )
    assert not (stored / "data.md").exists()


def test_the_report_checks_the_rules_of_the_gap_years_and_the_screening_window(
    stored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_command("ic_report", run_key="screen1", output=str(stored / "ic.md"), stdout=StringIO())
    seen: list[tuple[date, date]] = []
    monkeypatch.setattr(phase3_report, "_ruled", lambda start, end: seen.append((start, end)))

    call_command(
        "phase3_report",
        run_key="screen1",
        before=str(before_table(stored / "before.csv")),
        start="2016-01-01",
        end="2016-06-30",
        output=str(stored / "data.md"),
        appendix=str(stored / "appendix"),
        stdout=StringIO(),
    )

    run = ModelRun.objects.get(run_id="screen1.screening")
    assert seen == [(date(2016, 1, 1), date(2016, 6, 30)), (run.train_start, run.train_end)]


def test_gaps_with_a_known_cause_leave_the_acceptance(
    stored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = report(stored)
    monkeypatch.setattr(phase3_report, "observable_frame", from_table(known=True))

    explained = report(stored)

    assert (plain.loc[["fund_ep", "fund_rev_yoy"], "delisted"] > 0).all()
    assert explained.loc[CHECKED, ["current", "delisted"]].eq(0.0).all().all()
    pd.testing.assert_series_equal(
        explained.loc[CHECKED, "observable_delisted"],
        plain.loc[CHECKED, "delisted"],
        check_names=False,
    )
    assert plain.loc[CHECKED, "observable_delisted"].eq(0.0).all()


def test_the_report_refuses_partitions_that_the_database_no_longer_gives(
    stored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = FeatureDaily.objects.filter(stock_id=STOCKS[1], date__month=5)
    assert outside.update(in_universe=False, fund_ep=None) > 0
    monkeypatch.setattr(phase3_report, "observable_frame", from_table(known=True, drift="outside"))
    report(stored)
    monkeypatch.setattr(phase3_report, "observable_frame", from_table(known=True, drift="all"))

    with pytest.raises(CommandError, match="no longer gives"):
        report(stored)
    monkeypatch.setattr(phase3_report, "observable_frame", from_table(known=True, drift="emptied"))

    with pytest.raises(CommandError, match="no longer gives"):
        report(stored)


def test_gaps_in_masked_months_count_as_having_a_known_cause(
    stored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = report(stored)
    monkeypatch.setattr(phase3_report, "MASKED_MONTHS", {"fund_ep": frozenset({"2016-03"})})

    masked = report(stored)

    def share(table: pd.DataFrame, name: str, column: str = "delisted") -> float:
        value: float = table[column].loc[name]
        return value

    assert share(plain, "fund_ep") > 0
    assert share(masked, "fund_ep") == 0.0
    assert share(masked, "fund_ep", "observable_delisted") == share(plain, "fund_ep")
    assert share(masked, "fund_rev_yoy") == share(plain, "fund_rev_yoy")
