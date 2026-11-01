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
def test_moves_past_the_limit_are_jumps_and_event_days_are_flagged() -> None:
    flat = [10.0] * 6
    _series("1101", date(2014, 3, 1), [*flat, 10.8])
    _series("1216", date(2014, 3, 1), [10.0, 10.0, 12.0])
    _series("2330", date(2016, 3, 1), [*flat, 10.9])
    _series("2603", date(2016, 3, 1), [*flat, 11.005])
    _series("2884", date(2016, 3, 1), [*flat, 13.0])
    DividendEvent.objects.create(stock_id="2884", ex_date=date(2016, 3, 7), adj_factor=0.8)
    _series("3008", date(2016, 3, 1), [10.0, 15.0, *flat])
    _series("6488", date(2016, 3, 1), [20.0] * 6 + [10.0], factors=[0.5] * 6 + [1.0])
    _series("2412", date(2016, 3, 1), [*flat, 8.0])
    jumps = price_jumps(date(2010, 1, 1), free_days=5, tolerance=0.001)
    assert [(jump.stock_id, jump.day, jump.event) for jump in jumps] == [
        ("2884", date(2016, 3, 7), True),
        ("1216", date(2014, 3, 3), False),
        ("2412", date(2016, 3, 7), False),
        ("1101", date(2014, 3, 7), False),
    ]
    assert [jump.change for jump in jumps] == pytest.approx([0.3, 0.2, -0.2, 0.08])
    assert [jump.stock_id for jump in price_jumps(date(2015, 1, 1), 5, 0.001)] == [
        "2884",
        "2412",
    ]


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
    assert "| disposition_event | 0 | 0 | - | - |" in text
    assert "2 in stock_master, 1 with daily_price rows" in text
    assert "### Stocks: 1" in text
    assert "| 2330 | 2010-01-10 | 10.0000 | 12.0000 | +20.00% |  |" in text
    assert "### Funds (codes starting 00, some without a price limit): 0" in text
    row = "| daily_price | 2010-01-01..2026-10-30 | 1, 2 | failed | no | - | 3 | 14 | 22 |"
    assert f"| finmind {row}" in text
    assert "| finmind | dividend_event | - | - | not run |" in text
    assert "- daily_price 0001: x" in text


@pytest.mark.django_db
def test_jumps_before_since_are_left_out(tmp_path: Path) -> None:
    _series("2330", date(2010, 1, 4), [10.0] * 6 + [12.0])
    call_command("backfill_report", "--since", "2011-01-01", "--output", str(tmp_path / "r.md"))
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "### Stocks: 0" in text
    assert "| stock | date | previous |" not in text


def _backfill_run(
    run_id: int, selection: str | list[str], rejected: dict[str, str], **extra: object
) -> None:
    EtlRun.objects.create(
        run_id=run_id,
        source="finmind",
        dataset="daily_chip",
        start_date=date(2010, 1, 1),
        end_date=date(2026, 10, 30),
        status="failed",
        rows=5,
        checkpoint={"kind": "backfill", "selection": selection, "rejected": rejected, **extra},
    )


@pytest.mark.django_db
def test_retries_clear_the_rejections_they_loaded(tmp_path: Path) -> None:
    _backfill_run(1, "all", {"0001": "a", "0002": "b"}, finished=True, last_stock_id="9999")
    _backfill_run(2, ["0001", "0002"], {"0002": "still bad"}, finished=True)
    _backfill_run(3, ["0002"], {}, last_stock_id=None)
    EtlRun.objects.create(
        run_id=4,
        source="finmind",
        dataset="daily_valuation",
        start_date=date(2010, 1, 1),
        end_date=date(2026, 10, 30),
        checkpoint={"kind": "backfill", "selection": ["2330"]},
    )
    EtlRun.objects.create(
        run_id=5,
        source="finmind",
        dataset="daily_chip",
        start_date=date(2010, 1, 1),
        end_date=date(2026, 10, 30),
        checkpoint={"selection": "all", "last_date": "2026-10-30"},
    )
    call_command("backfill_report", "--output", str(tmp_path / "r.md"))
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    row = "| daily_chip | 2010-01-01..2026-10-30 | 1 | failed | yes | 9999 | - | 5 | 1 |"
    assert f"| finmind {row}" in text
    assert "- daily_chip 0002: still bad" in text
    assert "0001: a" not in text
    assert "| finmind | daily_valuation | - | - | not run |" in text


@pytest.mark.django_db
def test_many_fund_jumps_are_cut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.management.commands.backfill_report.FUND_ROWS", 1)
    _series("0050", date(2010, 1, 4), [10.0, 12.0, 15.0])
    call_command("backfill_report", "--output", str(tmp_path / "r.md"))
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "### Funds (codes starting 00, some without a price limit): 2" in text
    assert "| 0050 | 2010-01-06 |" in text
    assert "1 more fund rows not listed" in text
