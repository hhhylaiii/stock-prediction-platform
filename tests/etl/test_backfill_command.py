import os
import signal
from datetime import date
from typing import Any, ClassVar

import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.adapters import ADAPTERS
from apps.etl.adapters.base import BaseAdapter
from apps.etl.errors import SourceRequestError
from apps.etl.models import EtlRun, RawRecord, RawTwse
from apps.etl.upsert import upsert
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, Market, StockMaster

START = date(2010, 1, 1)
END = date(2010, 12, 31)


class StockEchoAdapter(BaseAdapter):
    """Returns one row per requested stock so the per-stock workflow runs without HTTP."""

    source = "echo"
    raw_model: ClassVar[type[RawRecord]] = RawTwse
    datasets = frozenset({"daily_price"})
    per_stock = True
    fail_on: ClassVar[str | None] = None
    interrupt_on: ClassVar[str | None] = None
    terminate_on: ClassVar[str | None] = None
    calls: ClassVar[list[tuple[str, date, date]]] = []

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        assert stock_id is not None
        self.calls.append((stock_id, start, end))
        if stock_id == self.fail_on:
            raise SourceRequestError(f"boom on {stock_id}")
        if stock_id == self.interrupt_on:
            raise KeyboardInterrupt
        if stock_id == self.terminate_on:
            os.kill(os.getpid(), signal.SIGTERM)
        return pd.DataFrame([{"stock_id": stock_id, "date": start, "volume": 1, "turnover": 1}])

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        return raw_df

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        return upsert(DailyPrice, frame, ("stock_id", "date"), ("volume", "turnover"))


@pytest.fixture(autouse=True)
def echo_adapter(monkeypatch: pytest.MonkeyPatch, october_2026: TradingDays, settings: Any) -> None:
    settings.ETL_RATE_LIMITS = {
        **settings.ETL_RATE_LIMITS,
        "echo": {
            "min_interval": 0.0,
            "max_retries": 0,
            "backoff_base": 0.0,
            "backoff_max": 0.0,
            "cooldown": 0.0,
        },
    }
    monkeypatch.setitem(ADAPTERS, "echo", StockEchoAdapter)
    monkeypatch.setattr(StockEchoAdapter, "calls", [])
    monkeypatch.setattr(StockEchoAdapter, "fail_on", None)
    monkeypatch.setattr(StockEchoAdapter, "interrupt_on", None)
    monkeypatch.setattr(StockEchoAdapter, "terminate_on", None)
    monkeypatch.setattr("apps.etl.adapters.TradingDays.from_db", lambda: october_2026)


@pytest.fixture
def stocks(db: None) -> None:
    for stock_id, delisted in [
        ("2330", None),
        ("1101", None),
        ("2317", None),
        ("3454", date(2026, 3, 27)),
        ("9999", date(2009, 12, 31)),
    ]:
        StockMaster.objects.create(
            stock_id=stock_id, name=stock_id, market=Market.TWSE, delist_date=delisted
        )


def _backfill(*extra: str) -> None:
    call_command(
        "backfill",
        "--source",
        "echo",
        "--dataset",
        "daily_price",
        "--from",
        "2010-01-01",
        "--to",
        "2010-12-31",
        *extra,
    )


def _fetched() -> list[str]:
    return [stock_id for stock_id, _, _ in StockEchoAdapter.calls]


@pytest.mark.usefixtures("stocks")
def test_backfill_walks_every_stock_alive_in_the_range() -> None:
    _backfill()
    assert _fetched() == ["1101", "2317", "2330", "3454"]
    assert StockEchoAdapter.calls[0][1:] == (START, END)
    assert DailyPrice.objects.count() == 4
    run = EtlRun.objects.get()
    assert (run.source, run.dataset, run.status, run.rows) == ("echo", "daily_price", "success", 4)
    assert run.checkpoint["kind"] == "backfill"
    assert run.checkpoint["selection"] == "all"
    assert (run.checkpoint["stocks"], run.checkpoint["last_stock_id"]) == (4, "3454")
    assert (run.start_date, run.end_date) == (START, END)


@pytest.mark.usefixtures("stocks")
def test_failure_keeps_the_last_finished_stock() -> None:
    StockEchoAdapter.fail_on = "2317"
    with pytest.raises(CommandError, match="boom on 2317"):
        _backfill()
    run = EtlRun.objects.get()
    assert (run.status, run.checkpoint["last_stock_id"], run.rows) == ("failed", "1101", 1)
    assert list(DailyPrice.objects.values_list("stock_id", flat=True)) == ["1101"]


@pytest.mark.usefixtures("stocks")
def test_resume_continues_after_the_checkpoint() -> None:
    StockEchoAdapter.fail_on = "2317"
    with pytest.raises(CommandError):
        _backfill()
    StockEchoAdapter.fail_on = None
    StockEchoAdapter.calls.clear()
    _backfill("--resume")
    assert _fetched() == ["2317", "2330", "3454"]
    resumed = EtlRun.objects.order_by("run_id").last()
    assert resumed is not None
    assert (resumed.status, resumed.checkpoint["last_stock_id"]) == ("success", "3454")
    assert DailyPrice.objects.count() == 4


@pytest.mark.usefixtures("stocks")
def test_interrupted_run_can_be_resumed() -> None:
    StockEchoAdapter.interrupt_on = "2330"
    with pytest.raises(KeyboardInterrupt):
        _backfill()
    assert EtlRun.objects.get().checkpoint["last_stock_id"] == "2317"
    StockEchoAdapter.interrupt_on = None
    StockEchoAdapter.calls.clear()
    _backfill("--resume")
    assert _fetched() == ["2330", "3454"]


@pytest.mark.usefixtures("stocks")
def test_resume_after_a_failure_on_the_first_stock_starts_over() -> None:
    StockEchoAdapter.fail_on = "1101"
    with pytest.raises(CommandError):
        _backfill()
    StockEchoAdapter.fail_on = None
    StockEchoAdapter.calls.clear()
    _backfill("--resume")
    assert _fetched() == ["1101", "2317", "2330", "3454"]


@pytest.mark.usefixtures("stocks")
def test_resume_needs_an_unfinished_run_with_the_same_range() -> None:
    _backfill()
    with pytest.raises(CommandError, match="no unfinished"):
        _backfill("--resume")
    StockEchoAdapter.fail_on = "2317"
    with pytest.raises(CommandError):
        call_command(
            "backfill", "--source", "echo", "--dataset", "daily_price", "--from", "2011-01-01"
        )
    with pytest.raises(CommandError, match="no unfinished"):
        _backfill("--resume")


@pytest.mark.usefixtures("stocks")
def test_stock_option_limits_the_run() -> None:
    _backfill("--stock", "2330", "--stock", "1101")
    assert _fetched() == ["1101", "2330"]


@pytest.mark.usefixtures("stocks")
def test_unknown_stock_is_a_command_error() -> None:
    with pytest.raises(CommandError, match="not in stock_master"):
        _backfill("--stock", "0000")


@pytest.mark.django_db
def test_empty_stock_master_is_a_command_error() -> None:
    with pytest.raises(CommandError, match="load_stock_master"):
        _backfill()


@pytest.mark.usefixtures("stocks")
def test_default_end_is_yesterday(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "apps.etl.management.commands.backfill.timezone.localdate", lambda: date(2026, 11, 1)
    )
    call_command("backfill", "--source", "echo", "--dataset", "daily_price", "--from", "2026-10-01")
    assert StockEchoAdapter.calls[0][1:] == (date(2026, 10, 1), date(2026, 10, 31))


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--from", "2010-12-31", "--to", "2010-01-01"], "after --to"),
        (["--from", "2010/01/01"], "ISO date"),
        (["--from", "2010-01-01", "--dataset", "nope"], "no dataset"),
    ],
)
def test_bad_arguments_are_command_errors(args: list[str], message: str, stocks: Any) -> None:
    options = ["--source", "echo", "--dataset", "daily_price", *args]
    with pytest.raises(CommandError, match=message):
        call_command("backfill", *options)


@pytest.mark.django_db
def test_unknown_source_is_a_command_error() -> None:
    with pytest.raises(CommandError, match="unknown source"):
        call_command("backfill", "--source", "nope", "--dataset", "x", "--from", "2010-01-01")


@pytest.mark.usefixtures("stocks")
def test_a_subset_run_does_not_steal_the_resume_point() -> None:
    StockEchoAdapter.fail_on = "2317"
    with pytest.raises(CommandError):
        _backfill()
    StockEchoAdapter.fail_on = "3454"
    with pytest.raises(CommandError):
        _backfill("--stock", "2330", "--stock", "3454")
    StockEchoAdapter.fail_on = None
    StockEchoAdapter.calls.clear()
    _backfill("--resume")
    assert _fetched() == ["2317", "2330", "3454"]


@pytest.mark.usefixtures("stocks")
def test_a_subset_run_resumes_its_own_selection() -> None:
    StockEchoAdapter.fail_on = "3454"
    with pytest.raises(CommandError):
        _backfill("--stock", "3454", "--stock", "2330")
    StockEchoAdapter.fail_on = None
    StockEchoAdapter.calls.clear()
    _backfill("--stock", "2330", "--stock", "3454", "--resume")
    assert _fetched() == ["3454"]


@pytest.mark.usefixtures("stocks")
def test_fetch_runs_with_the_same_range_are_not_resumed() -> None:
    EtlRun.objects.create(
        source="echo",
        dataset="daily_price",
        start_date=START,
        end_date=END,
        status=EtlRun.Status.FAILED,
        checkpoint={"last_date": "2010-06-30"},
    )
    with pytest.raises(CommandError, match="no unfinished"):
        _backfill("--resume")


@pytest.mark.usefixtures("stocks")
def test_a_run_still_marked_running_is_not_resumed() -> None:
    EtlRun.objects.create(
        source="echo",
        dataset="daily_price",
        start_date=START,
        end_date=END,
        checkpoint={
            "kind": "backfill",
            "selection": "all",
            "stocks": 4,
            "digest": "x",
            "last_stock_id": "1101",
        },
    )
    with pytest.raises(CommandError, match="still marked running"):
        _backfill("--resume")
    assert StockEchoAdapter.calls == []


@pytest.mark.usefixtures("stocks")
def test_a_resume_that_fails_at_once_keeps_the_original_checkpoint() -> None:
    StockEchoAdapter.fail_on = "2317"
    with pytest.raises(CommandError):
        _backfill()
    with pytest.raises(CommandError):
        _backfill("--resume")
    StockEchoAdapter.fail_on = None
    StockEchoAdapter.calls.clear()
    _backfill("--resume")
    assert _fetched() == ["2317", "2330", "3454"]


@pytest.mark.usefixtures("stocks")
def test_termination_marks_the_run_failed_so_it_can_resume() -> None:
    previous = signal.getsignal(signal.SIGTERM)
    StockEchoAdapter.terminate_on = "2330"
    with pytest.raises(SystemExit):
        _backfill()
    run = EtlRun.objects.get()
    assert (run.status, run.checkpoint["last_stock_id"]) == ("failed", "2317")
    assert signal.getsignal(signal.SIGTERM) == previous
    StockEchoAdapter.terminate_on = None
    StockEchoAdapter.calls.clear()
    _backfill("--resume")
    assert _fetched() == ["2330", "3454"]


@pytest.mark.usefixtures("stocks")
def test_adapters_that_fetch_whole_market_days_cannot_backfill() -> None:
    with pytest.raises(CommandError, match="one stock at a time"):
        call_command(
            "backfill", "--source", "twse", "--dataset", "daily_price", "--from", "2010-01-04"
        )


def _run_on(monkeypatch: pytest.MonkeyPatch, today: date, *extra: str) -> None:
    monkeypatch.setattr("apps.etl.management.commands.backfill.timezone.localdate", lambda: today)
    call_command(
        "backfill", "--source", "echo", "--dataset", "daily_price", "--from", "2026-10-01", *extra
    )


@pytest.mark.usefixtures("stocks")
def test_resume_without_to_continues_the_run_started_on_an_earlier_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    StockEchoAdapter.fail_on = "2317"
    with pytest.raises(CommandError):
        _run_on(monkeypatch, date(2026, 11, 2))
    StockEchoAdapter.fail_on = None
    StockEchoAdapter.calls.clear()
    _run_on(monkeypatch, date(2026, 11, 3), "--resume")
    assert StockEchoAdapter.calls == [
        (code, date(2026, 10, 1), date(2026, 11, 1)) for code in ("2317", "2330")
    ]


@pytest.mark.usefixtures("stocks")
def test_resume_refuses_when_the_stock_list_changed() -> None:
    StockEchoAdapter.fail_on = "2317"
    with pytest.raises(CommandError):
        _backfill()
    StockMaster.objects.create(stock_id="2000", name="2000", market=Market.TWSE)
    StockEchoAdapter.fail_on = None
    StockEchoAdapter.calls.clear()
    with pytest.raises(CommandError, match="from 4 to 5 stocks"):
        _backfill("--resume")
    assert StockEchoAdapter.calls == []
