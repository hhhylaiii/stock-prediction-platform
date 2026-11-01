from datetime import date, timedelta
from pathlib import Path

import pytest
from django.core.management import call_command

from apps.etl.backfill_report import price_jumps
from apps.etl.models import EtlRun
from apps.market.models import DailyPrice, DividendEvent, Market, StockMaster


def _series(
    stock: str, start: date, closes: list[float], factors: list[float] | None = None
) -> None:
    DailyPrice.objects.bulk_create(
        DailyPrice(
            stock_id=stock,
            date=start + timedelta(days=offset),
            close=close,
            volume=1,
            turnover=1,
            adj_factor=1.0 if factors is None else factors[offset],
        )
        for offset, close in enumerate(closes)
    )


@pytest.mark.django_db
def test_only_unexplained_moves_past_the_limit_are_jumps() -> None:
    flat = [10.0] * 6
    _series("1101", date(2014, 3, 1), [*flat, 10.8])
    _series("2330", date(2016, 3, 1), [*flat, 10.9])
    _series("2884", date(2016, 3, 1), [*flat, 13.0])
    DividendEvent.objects.create(stock_id="2884", ex_date=date(2016, 3, 7), adj_factor=0.8)
    _series("3008", date(2016, 3, 1), [10.0, 15.0, *flat])
    _series("6488", date(2016, 3, 1), [20.0] * 6 + [10.0], factors=[0.5] * 6 + [1.0])
    _series("2412", date(2016, 3, 1), [*flat, 8.0])
    jumps = price_jumps(date(2010, 1, 1), free_days=5, tolerance=0.001)
    assert [(jump.stock_id, jump.day, jump.previous) for jump in jumps] == [
        ("2412", date(2016, 3, 7), 10.0),
        ("1101", date(2014, 3, 7), 10.0),
    ]
    assert [jump.change for jump in jumps] == pytest.approx([-0.2, 0.08])
    assert price_jumps(date(2015, 1, 1), free_days=5, tolerance=0.001)[0].stock_id == "2412"


@pytest.mark.django_db
def test_the_report_covers_tables_delisted_stocks_jumps_and_runs(tmp_path: Path) -> None:
    StockMaster.objects.create(stock_id="2330", name="台積電", market=Market.TWSE)
    StockMaster.objects.create(
        stock_id="5371", name="中光電", market=Market.TPEX, delist_date=date(2026, 9, 3)
    )
    StockMaster.objects.create(
        stock_id="9999", name="9999", market=Market.TWSE, delist_date=date(2009, 1, 5)
    )
    _series("2330", date(2010, 1, 4), [10.0] * 6 + [12.0])
    _series("5371", date(2010, 1, 4), [10.0] * 3)
    for run_id, status, selection, rejected in [
        (1, "failed", "all", {}),
        (2, "failed", "all", {f"{code:04d}": "x" for code in range(1, 23)}),
        (3, "success", ["2330"], {}),
    ]:
        EtlRun.objects.create(
            run_id=run_id,
            source="finmind",
            dataset="daily_price",
            start_date=date(2010, 1, 1),
            end_date=date(2026, 10, 30),
            status=status,
            rows=7,
            checkpoint={
                "kind": "backfill",
                "selection": selection,
                "stocks": 3,
                "rejected": rejected,
            },
        )
    output = tmp_path / "report.md"
    call_command("backfill_report", "--output", str(output))
    text = output.read_text(encoding="utf-8")
    assert "| daily_price | 10 | 2 | 2010-01-04 | 2010-01-10 |" in text
    assert "| weekly_holding | 0 | 0 | - | - |" in text
    assert "| trading_calendar | 0 | - | - | - |" in text
    assert "2 in stock_master, 1 with daily_price rows" in text
    assert "1 adjusted daily moves since 2010-01-01" in text
    assert "| 2330 | 2010-01-10 | 10.0000 | 12.0000 | +20.00% |" in text
    assert (
        "| finmind | daily_price | 2 | failed | 2010-01-01..2026-10-30 | 3 | 7 | 22 (0001" in text
    )
    assert "0020 …" in text


@pytest.mark.django_db
def test_jumps_before_since_are_left_out(tmp_path: Path) -> None:
    _series("2330", date(2010, 1, 4), [10.0] * 6 + [12.0])
    call_command("backfill_report", "--since", "2011-01-01", "--output", str(tmp_path / "r.md"))
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "0 adjusted daily moves since 2011-01-01" in text
    assert "| stock | date | previous |" not in text
