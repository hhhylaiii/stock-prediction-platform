import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection

from apps.market.calendar import TradingDays
from apps.market.models import TradingCalendar
from apps.research import lock
from apps.research.lock import LOCK_ID, ResearchBusyError, research_lock
from apps.research.models import ModelRun, Prediction
from apps.research.monitor import run_monitor
from apps.research.predict import predict
from apps.research.promotion import promote, restore

DAY = date(2026, 10, 28)
WAIT = 10


@pytest.fixture
def calendar_rows(db: None) -> None:
    first = date(2026, 1, 1)
    days = (first + timedelta(days=n) for n in range(365))
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5) for day in days
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
    return seen == [True]


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
        with research_lock("predict 2026-10-28"):
            assert holds_here()
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
    def label() -> str:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_setting('application_name')")
            (name,) = cursor.fetchone()
        return str(name)

    before = label()
    with research_lock("restore r1"):
        assert "restore r1" in label()
    assert label() == before


@pytest.mark.django_db
def test_predict_monitor_promote_and_restore_refuse_while_another_job_holds_the_lock(
    calendar_rows: None, tmp_path: Path
) -> None:
    calendar = TradingDays.from_db()
    with held_elsewhere("research.step features 2026-10-28"):
        for call in (
            lambda: predict(DAY, tmp_path),
            lambda: run_monitor(DAY, calendar),
            lambda: promote("prod.rank", DAY, tmp_path),
            lambda: restore("r1", DAY, "back"),
        ):
            with pytest.raises(ResearchBusyError, match=r"research\.step features 2026-10-28"):
                call()
    assert not ModelRun.objects.exists()
    assert not Prediction.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("command", "options"),
    [
        ("predict", {"date": "2026-10-28"}),
        ("monitor", {"date": "2026-10-28"}),
        ("promote", {"run_key": "prod.rank", "date": "2026-10-28"}),
        ("restore_run", {"run_id": "r1", "date": "2026-10-28", "reason": "back"}),
    ],
)
def test_the_commands_report_who_holds_the_lock(
    calendar_rows: None, command: str, options: dict[str, str]
) -> None:
    with held_elsewhere("monitor 2026-10-27"), pytest.raises(CommandError) as caught:
        call_command(command, **options)
    assert "monitor 2026-10-27" in str(caught.value)
    assert "predict, monitor, promote, restore" in str(caught.value)
