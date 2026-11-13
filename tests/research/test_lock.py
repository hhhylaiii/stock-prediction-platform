import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command, load_command_class
from django.core.management.base import CommandError
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, Market, StockMaster, TradingCalendar
from apps.research import lock
from apps.research.feature_coverage import run_coverage
from apps.research.lock import LOCK_ID, ResearchBusyError, locked, research_lock
from apps.research.management.commands import build_features, build_labels
from apps.research.models import FeatureDaily, LabelDaily
from apps.research.monitor import run_monitor
from apps.research.predict import predict
from apps.research.promotion import promote, restore

DAY = date(2026, 10, 28)
WAIT = 10
BUILDS = [("build_features", {"date": DAY}), ("build_labels", {"start": DAY, "end": DAY})]


@pytest.fixture
def calendar_rows(db: None) -> None:
    first = date(2026, 1, 1)
    days = (first + timedelta(days=n) for n in range(365))
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5) for day in days
    )


@pytest.fixture
def one_bar(calendar_rows: None, settings: Any, tmp_path: Path) -> None:
    settings.FEATURE_PARQUET_DIR = tmp_path
    StockMaster.objects.create(
        stock_id="2330", name="台積電", market=Market.TWSE, industry="半導體業"
    )
    DailyPrice.objects.create(
        stock_id="2330",
        date=DAY,
        open=100,
        high=101,
        low=99,
        close=100,
        volume=1000,
        turnover=100000,
    )


def holds_here() -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND objid = %s "
            "AND granted AND pid = pg_backend_pid()",
            [LOCK_ID],
        )
        (count,) = cursor.fetchone()
    return bool(count)


def label() -> str:
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_setting('application_name')")
        (name,) = cursor.fetchone()
    return str(name)


@contextmanager
def held_elsewhere(holder: str) -> Iterator[None]:
    """Another session holds the research lock until the block ends."""
    acquired, release = threading.Event(), threading.Event()

    def hold() -> None:
        try:
            with research_lock(holder):
                acquired.set()
                release.wait(WAIT)
        finally:
            connection.close()

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(WAIT)
    try:
        yield
    finally:
        release.set()
        thread.join(WAIT)


def free_elsewhere() -> bool:
    """Whether another session can take the research lock right now."""
    seen: list[bool] = []

    def take() -> None:
        try:
            with research_lock("probe"):
                seen.append(True)
        except ResearchBusyError:
            seen.append(False)
        finally:
            connection.close()

    thread = threading.Thread(target=take)
    thread.start()
    thread.join(WAIT)
    assert not thread.is_alive()
    assert len(seen) == 1
    return seen[0]


@pytest.mark.django_db
def test_the_lock_is_held_inside_the_block_and_released_after_it() -> None:
    with research_lock("predict 2026-10-28"):
        assert holds_here()
        assert not free_elsewhere()
    assert not holds_here()
    assert free_elsewhere()


@pytest.mark.django_db
def test_the_lock_is_reentrant_in_one_session_and_kept_until_the_outer_block_ends() -> None:
    with research_lock("promote prod.rank"):
        with CaptureQueriesContext(connection) as inner, research_lock("predict 2026-10-28"):
            pass
        assert inner.captured_queries == []
        assert "promote prod.rank" in label()
        assert holds_here()
        assert not free_elsewhere()
    assert not holds_here()


@pytest.mark.django_db
def test_an_error_inside_the_block_releases_the_lock() -> None:
    with pytest.raises(ValueError, match="boom"), research_lock("monitor 2026-10-28"):
        raise ValueError("boom")
    assert not holds_here()
    assert free_elsewhere()


@pytest.mark.django_db
def test_a_busy_lock_names_its_holder_and_why_jobs_wait() -> None:
    with (
        held_elsewhere("monitor 2026-10-28"),
        pytest.raises(ResearchBusyError) as caught,
        research_lock("predict 2026-10-28"),
    ):
        pytest.fail("the lock was taken twice")
    message = str(caught.value)
    assert "monitor 2026-10-28" in message
    assert "pid " in message
    assert "predict, monitor, promote, restore" in message
    assert not holds_here()


@pytest.mark.django_db
def test_a_locked_function_is_labelled_with_its_job_its_argument_and_the_time() -> None:
    acquired, release = threading.Event(), threading.Event()

    @locked("predict")
    def waits(day: date) -> None:
        acquired.set()
        release.wait(WAIT)

    def hold() -> None:
        try:
            waits(DAY)
        finally:
            connection.close()

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(WAIT)
    try:
        with pytest.raises(ResearchBusyError) as caught, research_lock("monitor 2026-10-28"):
            pytest.fail("the lock was taken twice")
    finally:
        release.set()
        thread.join(WAIT)
    held = r"'research \d{2}-\d{2} \d{2}:\d{2} predict 2026-10-28' \(pid \d+\)"
    assert re.search(held, str(caught.value))


@pytest.mark.django_db(transaction=True)
def test_a_failed_release_closes_the_connection_so_the_lock_goes_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lock, "UNLOCK_SQL", "SELECT no_such_function(%s, %s)")
    with research_lock("predict 2026-10-28"):
        assert holds_here()
    assert connection.connection is None
    assert free_elsewhere()


@pytest.mark.django_db
def test_the_holder_label_is_restored_once_the_lock_is_released() -> None:
    before = label()
    with research_lock("restore r1"):
        assert "restore r1" in label()
    assert label() == before


@pytest.mark.django_db
def test_the_locked_jobs_refuse_while_another_job_holds_the_lock(
    calendar_rows: None, tmp_path: Path
) -> None:
    calendar = TradingDays.from_db()
    with held_elsewhere("features 2026-10-28"):
        for call in (
            lambda: predict(DAY, tmp_path),
            lambda: run_monitor(DAY, calendar),
            lambda: promote("prod.rank", DAY, tmp_path),
            lambda: restore("r1", DAY, "back"),
            lambda: run_coverage(DAY, calendar, tmp_path),
        ):
            with pytest.raises(ResearchBusyError, match="features 2026-10-28"):
                call()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("command", "options"),
    [
        ("predict", {"date": "2026-10-28"}),
        ("monitor", {"date": "2026-10-28"}),
        ("promote", {"run_key": "prod.rank", "date": "2026-10-28"}),
        ("restore_run", {"run_id": "r1", "date": "2026-10-28", "reason": "back"}),
        ("feature_coverage", {"date": DAY}),
    ],
)
def test_the_commands_report_who_holds_the_lock(
    calendar_rows: None, command: str, options: dict[str, Any]
) -> None:
    with held_elsewhere("monitor 2026-10-27"), pytest.raises(CommandError) as caught:
        call_command(command, **options)
    assert "monitor 2026-10-27" in str(caught.value)
    assert "predict, monitor, promote, restore" in str(caught.value)


@pytest.mark.django_db
@pytest.mark.parametrize(("command", "options"), BUILDS)
def test_a_manual_build_fails_at_once_while_another_job_holds_the_lock(
    one_bar: None, command: str, options: dict[str, Any]
) -> None:
    with held_elsewhere("predict 2026-10-27"), pytest.raises(CommandError) as caught:
        call_command(command, **options)
    assert "predict 2026-10-27" in str(caught.value)
    assert "feature and label builds" in str(caught.value)
    assert not FeatureDaily.objects.exists()
    assert not LabelDaily.objects.exists()
    assert not holds_here()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("command", "module", "writer"),
    [
        ("build_features", build_features, "write_features"),
        ("build_labels", build_labels, "write_labels"),
    ],
)
def test_a_manual_build_holds_the_lock_while_it_writes_and_names_its_range(
    one_bar: None, monkeypatch: pytest.MonkeyPatch, command: str, module: Any, writer: str
) -> None:
    seen: list[tuple[bool, bool, str]] = []
    real = getattr(module, writer)

    def spy(*args: Any) -> int:
        seen.append((holds_here(), free_elsewhere(), label()))
        return int(real(*args))

    monkeypatch.setattr(module, writer, spy)
    call_command(command, start=DAY, end=DAY)
    assert len(seen) == 1
    held, free, name = seen[0]
    assert held
    assert not free
    assert re.search(rf"research \d{{2}}-\d{{2}} \d{{2}}:\d{{2}} {command} {DAY}\.\.{DAY}$", name)
    assert not holds_here()
    assert free_elsewhere()


@pytest.mark.django_db
@pytest.mark.parametrize(("command", "options"), BUILDS)
def test_a_build_inside_a_scheduled_step_runs_under_the_step_lock(
    one_bar: None, command: str, options: dict[str, Any]
) -> None:
    with research_lock("recent_features 2026-10-28"):
        call_command(command, **options)
        assert "recent_features 2026-10-28" in label()
        assert holds_here()
    assert not holds_here()


@pytest.mark.django_db
def test_a_build_with_a_bad_argument_says_so_even_while_the_lock_is_held(one_bar: None) -> None:
    saturday = date(2026, 10, 31)
    with held_elsewhere("predict 2026-10-27"), pytest.raises(CommandError) as caught:
        call_command("build_features", date=saturday)
    assert "is not a trading day" in str(caught.value)


@pytest.mark.parametrize("command", ["build_features", "build_labels"])
def test_the_build_commands_say_in_their_help_that_they_take_the_lock(command: str) -> None:
    parser = load_command_class("apps.research", command).create_parser("manage.py", command)
    assert "research lock" in parser.format_help()
