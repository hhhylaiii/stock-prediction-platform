from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import redis
import responses
from celery import Task
from celery.signals import task_failure
from django.apps import apps
from django.core.management.base import CommandError
from django.db import DatabaseError, InterfaceError, OperationalError
from django.test import override_settings

from apps.etl import task_alerts, tasks
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.quality import CHECKS, should_run_prediction
from apps.etl.task_alerts import CHECK, KEY_TTL, on_task_failure, report_failure
from apps.market.calendar import OutsideCalendarError
from config.celery import app

WEBHOOK = "https://hooks.example.test/alerts"
TAIPEI = ZoneInfo("Asia/Taipei")
DAY = date(2026, 11, 12)
OFF_CALENDAR = "2027-01-02 is outside 2008-01-01..2026-12-31"

pytestmark = pytest.mark.django_db


@app.task(name="tests.boom")
def boom(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("/srv/app/secret/path is gone")


@app.task(name="tests.off_calendar")
def off_calendar() -> None:
    raise OutsideCalendarError(OFF_CALENDAR)


@app.task(name="tests.db_down")
def db_down() -> None:
    raise OperationalError('connection to server at "db" failed')


@app.task(name="tests.db_down_too")
def db_down_too() -> None:
    raise InterfaceError("connection already closed")


@app.task(name="tests.boom_too")
def boom_too() -> None:
    raise RuntimeError("another task, the same class")


@app.task(name="tests.either")
def either(kind: str) -> None:
    raise {"value": ValueError, "key": KeyError}[kind](kind)


@app.task(bind=True, name="tests.flaky", max_retries=2)
def flaky(self: Task[Any, Any], failures: int) -> str:
    if self.request.retries < failures:
        raise self.retry(exc=SourceNotReadyError("not published yet"), countdown=0)
    return "done"


@pytest.fixture(autouse=True)
def failures_are_traced() -> Iterator[None]:
    """Eager runs hand a failure to Celery's tracer, as a worker does, instead of raising it."""
    before = app.conf.CELERY_TASK_EAGER_PROPAGATES
    app.conf.CELERY_TASK_EAGER_PROPAGATES = False
    yield
    app.conf.CELERY_TASK_EAGER_PROPAGATES = before


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def _set(day: date, at: time = time(15, 10)) -> None:
        moment = datetime.combine(day, at, tzinfo=TAIPEI)
        monkeypatch.setattr("apps.etl.task_alerts.timezone.localtime", lambda: moment)

    _set(DAY)
    return _set


@pytest.fixture
def claimed(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """An in-memory stand-in for the Redis keys that de-duplicate alerts."""
    keys: set[str] = set()

    def claim(key: str) -> bool:
        first = key not in keys
        keys.add(key)
        return first

    monkeypatch.setattr("apps.etl.task_alerts._claim", claim)
    monkeypatch.setattr("apps.etl.task_alerts._release", keys.discard)
    return keys


@pytest.fixture
def hook() -> Any:
    with override_settings(ALERT_WEBHOOK_URL=WEBHOOK), responses.RequestsMock() as mock:
        mock.post(WEBHOOK, status=200)
        mock.assert_all_requests_are_fired = False
        yield mock


def _bodies(hook: Any) -> list[dict[str, Any]]:
    bodies = []
    for call in hook.calls:
        raw = call.request.body
        bodies.append(json.loads(raw.decode() if isinstance(raw, bytes) else str(raw)))
    return bodies


def _row(day: date = DAY) -> DataQualityLog:
    return DataQualityLog.objects.get(date=day, check_name=CHECK)


class Unprintable:
    """An argument whose repr raises."""

    def __repr__(self) -> str:
        raise RuntimeError("no repr")


@pytest.mark.usefixtures("clock", "claimed")
def test_an_unhandled_error_alerts_with_the_task_and_error_class_only(hook: Any) -> None:
    boom.apply(args=("night",), kwargs={"day": "2026-11-12"})
    (body,) = _bodies(hook)
    assert body == {
        "text": (
            "task failed: tests.boom (RuntimeError) args=('night',) kwargs={'day': '2026-11-12'}"
        ),
        "alerts": [
            {
                "date": "2026-11-12",
                "check": "task_failure",
                "severity": "warning",
                "detail": {
                    "task": "tests.boom",
                    "error": "RuntimeError",
                    "message": None,
                    "context": "args=('night',) kwargs={'day': '2026-11-12'}",
                    "first_at": "2026-11-12T15:10:00+08:00",
                },
            }
        ],
    }
    text = json.dumps(body)
    assert "secret" not in text
    assert "Traceback" not in text
    assert ".py" not in text


@pytest.mark.usefixtures("claimed")
def test_the_same_task_and_error_alert_once_a_day(hook: Any, clock: Callable[..., None]) -> None:
    boom.apply()
    boom.apply()
    assert len(hook.calls) == 1
    clock(date(2026, 11, 13))
    boom.apply()
    assert len(hook.calls) == 2
    assert [body["alerts"][0]["date"] for body in _bodies(hook)] == ["2026-11-12", "2026-11-13"]


@pytest.mark.usefixtures("clock", "claimed")
def test_another_task_or_another_error_class_is_another_alert(hook: Any) -> None:
    boom.apply()
    boom_too.apply()
    assert len(hook.calls) == 2
    either.apply(args=("value",))
    either.apply(args=("key",))
    either.apply(args=("key",))
    assert len(hook.calls) == 4


@pytest.mark.usefixtures("clock", "claimed")
def test_an_alert_that_did_not_arrive_is_posted_again_at_the_next_failure() -> None:
    with override_settings(ALERT_WEBHOOK_URL=WEBHOOK), responses.RequestsMock() as mock:
        mock.post(WEBHOOK, status=503)
        mock.post(WEBHOOK, status=200)
        boom.apply()
        boom.apply()
        boom.apply()
        assert len(mock.calls) == 2
    assert _row().detail["tasks"]["tests.boom"]["count"] == 3


@pytest.mark.usefixtures("clock", "claimed")
def test_a_calendar_error_carries_its_own_message(hook: Any) -> None:
    off_calendar.apply()
    (body,) = _bodies(hook)
    detail = body["alerts"][0]["detail"]
    assert (detail["error"], detail["message"]) == ("OutsideCalendarError", OFF_CALENDAR)
    assert body["text"] == (
        f"task failed: tests.off_calendar (OutsideCalendarError: {OFF_CALENDAR})"
        " args=() kwargs={}"
    )


@pytest.mark.usefixtures("clock", "claimed")
def test_long_arguments_and_messages_are_cut(hook: Any) -> None:
    boom.apply(args=("x" * 500,), kwargs={"note": "y" * 500})
    report_failure("tests.direct", OutsideCalendarError("z" * 500), "label")
    first, second = (body["alerts"][0]["detail"] for body in _bodies(hook))
    assert len(first["context"]) <= 420
    assert "x" * 190 in first["context"]
    assert "x" * 201 not in first["context"]
    assert "y" * 201 not in first["context"]
    assert len(second["message"]) == 200


@pytest.mark.usefixtures("clock", "claimed")
def test_a_retry_is_not_a_failure_but_running_out_of_them_is(hook: Any) -> None:
    flaky.apply(args=(2,))
    assert len(hook.calls) == 0
    assert not DataQualityLog.objects.exists()
    result = flaky.apply(args=(9,))
    assert result.failed()
    (body,) = _bodies(hook)
    assert body["alerts"][0]["detail"]["error"] == "SourceNotReadyError"
    assert _row().detail["tasks"]["tests.flaky"]["count"] == 1


@pytest.mark.usefixtures("clock", "claimed")
def test_a_dead_database_alerts_once_for_every_task_and_is_not_written_to(hook: Any) -> None:
    db_down.apply()
    db_down_too.apply()
    db_down.apply()
    (body,) = _bodies(hook)
    assert body["text"] == (
        "task failed: database error (OperationalError), first seen in tests.db_down "
        "args=() kwargs={}; one alert a day covers every task"
    )
    assert not DataQualityLog.objects.exists()


@pytest.mark.usefixtures("clock", "claimed", "hook")
def test_the_days_row_counts_each_tasks_failures() -> None:
    boom.apply()
    boom.apply()
    off_calendar.apply()
    row = _row()
    assert row.severity == "warning"
    assert row.detail == {
        "tasks": {
            "tests.boom": {
                "count": 2,
                "first_at": "2026-11-12T15:10:00+08:00",
                "errors": ["RuntimeError"],
            },
            "tests.off_calendar": {
                "count": 1,
                "first_at": "2026-11-12T15:10:00+08:00",
                "errors": ["OutsideCalendarError"],
            },
        }
    }
    assert DataQualityLog.objects.filter(check_name=CHECK).count() == 1


@pytest.mark.usefixtures("claimed", "hook")
def test_the_first_time_a_task_failed_is_kept(clock: Callable[..., None]) -> None:
    boom.apply()
    clock(DAY, time(21, 30))
    boom.apply()
    assert _row().detail["tasks"]["tests.boom"] == {
        "count": 2,
        "first_at": "2026-11-12T15:10:00+08:00",
        "errors": ["RuntimeError"],
    }


@pytest.mark.usefixtures("clock", "claimed")
def test_a_row_that_cannot_be_written_does_not_stop_the_alert(
    hook: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise DatabaseError("relation data_quality_log does not exist")

    monkeypatch.setattr("apps.etl.task_alerts._count", refuse)
    boom.apply()
    assert len(hook.calls) == 1
    assert "could not record: DatabaseError" in caplog.text
    assert "data_quality_log does not exist" not in caplog.text


@pytest.mark.usefixtures("clock", "claimed")
def test_the_handler_never_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(*args: Any) -> None:
        raise RuntimeError("posting broke")

    monkeypatch.setattr("apps.etl.task_alerts._post", broken)
    report_failure("tests.direct", ValueError("x"))
    on_task_failure(sender=None, exception=None, args=None, kwargs=None)
    assert "task failure alert broke: RuntimeError" in caplog.text


@pytest.mark.usefixtures("clock", "claimed")
def test_arguments_that_cannot_be_printed_do_not_break_the_handler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    on_task_failure(sender=None, exception=ValueError("x"), args=(Unprintable(),), kwargs=None)
    assert "task failure alert broke: RuntimeError" in caplog.text


@pytest.mark.usefixtures("clock", "claimed")
def test_without_a_webhook_the_failure_is_only_logged(caplog: pytest.LogCaptureFixture) -> None:
    with responses.RequestsMock() as mock, caplog.at_level(logging.ERROR):
        boom.apply(args=("night",))
        assert len(mock.calls) == 0
    assert "task failed tests.boom (RuntimeError)" in caplog.text
    assert _row().detail["tasks"]["tests.boom"]["count"] == 1


@pytest.mark.usefixtures("clock", "claimed")
def test_a_failing_webhook_is_logged_by_its_class(caplog: pytest.LogCaptureFixture) -> None:
    with override_settings(ALERT_WEBHOOK_URL=WEBHOOK), responses.RequestsMock() as mock:
        mock.post(WEBHOOK, status=500)
        boom.apply()
    assert "task failure alert webhook failed: HTTPError" in caplog.text
    assert WEBHOOK not in caplog.text


class FakeRedis:
    """Records what the claim asked of the store."""

    def __init__(self) -> None:
        self.keys: dict[str, str] = {}
        self.options: list[dict[str, Any]] = []
        self.closed = 0

    def set(self, key: str, value: str, **options: Any) -> bool | None:
        self.options.append(options)
        if key in self.keys:
            return None
        self.keys[key] = value
        return True

    def delete(self, key: str) -> int:
        return int(self.keys.pop(key, None) is not None)

    def close(self) -> None:
        self.closed += 1


def test_the_first_claim_of_a_key_wins_and_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeRedis()
    urls: list[tuple[str, dict[str, Any]]] = []

    def connect(url: str, **options: Any) -> FakeRedis:
        urls.append((url, options))
        return store

    monkeypatch.setattr("apps.etl.task_alerts.redis.Redis.from_url", connect)
    with override_settings(TASK_ALERT_REDIS_URL="redis://store.test:6379/3"):
        assert task_alerts._claim("alert:task_failure:2026-11-12:etl.run_batch:ValueError")
        assert not task_alerts._claim("alert:task_failure:2026-11-12:etl.run_batch:ValueError")
    assert urls[0] == (
        "redis://store.test:6379/3",
        {"socket_connect_timeout": 2.0, "socket_timeout": 2.0},
    )
    assert store.options == [{"nx": True, "ex": KEY_TTL}] * 2
    assert KEY_TTL == 36 * 3600
    assert store.closed == 2


def test_a_released_key_can_be_claimed_again(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeRedis()
    monkeypatch.setattr("apps.etl.task_alerts.redis.Redis.from_url", lambda url, **options: store)
    with override_settings(TASK_ALERT_REDIS_URL="redis://store.test:6379/3"):
        assert task_alerts._claim("key")
        task_alerts._release("key")
        assert task_alerts._claim("key")
    assert store.closed == 3


def test_a_store_that_cannot_answer_never_silences_an_alert(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def unreachable(url: str, **options: Any) -> None:
        raise redis.ConnectionError("Error 111 connecting to store.test:6379")

    monkeypatch.setattr("apps.etl.task_alerts.redis.Redis.from_url", unreachable)
    with override_settings(TASK_ALERT_REDIS_URL="redis://store.test:6379/3"):
        assert task_alerts._claim("key")
        assert task_alerts._claim("key")
        task_alerts._release("key")
    assert "could not reach its store: ConnectionError" in caplog.text
    assert "store.test" not in caplog.text


def test_without_a_store_nothing_is_de_duplicated_or_connected(
    monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    def forbidden(url: str, **options: Any) -> None:
        raise AssertionError("the tests must not reach Redis")

    monkeypatch.setattr("apps.etl.task_alerts.redis.Redis.from_url", forbidden)
    assert settings.TASK_ALERT_REDIS_URL == ""
    assert task_alerts._claim("key")
    assert task_alerts._claim("key")
    task_alerts._release("key")


def test_connecting_the_signal_is_idempotent_and_touches_nothing(
    monkeypatch: pytest.MonkeyPatch, django_assert_num_queries: Any
) -> None:
    def forbidden(url: str, **options: Any) -> None:
        raise AssertionError("ready() must not reach Redis")

    monkeypatch.setattr("apps.etl.task_alerts.redis.Redis.from_url", forbidden)
    with (
        override_settings(TASK_ALERT_REDIS_URL="redis://store.test:6379/3"),
        responses.RequestsMock() as mock,
        django_assert_num_queries(0),
    ):
        apps.get_app_config("etl").ready()
        apps.get_app_config("etl").ready()
        assert len(mock.calls) == 0
    ours = [
        receiver for receiver in task_failure.receivers if receiver[0][0] == "apps.etl.task_alerts"
    ]
    assert len(ours) == 1


@pytest.mark.usefixtures("clock", "claimed", "hook")
def test_a_failure_row_never_moves_the_research_gate() -> None:
    for source, dataset in tasks._daily_jobs():
        EtlRun.objects.create(
            source=source, dataset=dataset, start_date=DAY, end_date=DAY, status="success"
        )
    for check in CHECKS:
        DataQualityLog.objects.create(date=DAY, check_name=check, severity="info", detail={})
    boom.apply()
    assert _row().severity == "warning"
    assert CHECK not in {*CHECKS, *tasks.AUDITED}
    assert (tasks.etl_complete(DAY), should_run_prediction(DAY), tasks.ready_for_research(DAY)) == (
        True,
        True,
        True,
    )


class Raising:
    """An adapter whose run raises the given error."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def run(self, dataset: str, start: date, end: date) -> EtlRun:
        raise self.error


@pytest.mark.usefixtures("clock", "claimed")
def test_an_unexpected_error_a_task_swallows_is_alerted_under_the_tasks_name(
    hook: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "apps.etl.tasks.build_adapter", lambda source, calendar=None: Raising(ValueError("bug"))
    )
    for dataset in ("daily_price", "index_daily"):
        result = tasks.fetch_day.apply(args=("twse", dataset, "2026-10-28"))
        assert (result.successful(), result.get()) == (True, "failed")
    (body,) = _bodies(hook)
    assert body["text"] == "task failed: etl.fetch_day (ValueError) twse/daily_price 2026-10-28"
    assert body["alerts"][0]["detail"]["context"] == "twse/daily_price 2026-10-28"
    assert _row().detail["tasks"]["etl.fetch_day"]["count"] == 2


@pytest.mark.usefixtures("clock", "claimed")
def test_a_source_error_a_task_swallows_stays_with_the_evening_audit(
    hook: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "apps.etl.tasks.build_adapter",
        lambda source, calendar=None: Raising(SchemaChangedError("fields changed")),
    )
    result = tasks.fetch_day.apply(args=("twse", "daily_price", "2026-10-28"))
    assert result.get() == "failed"
    assert len(hook.calls) == 0
    assert not DataQualityLog.objects.filter(check_name=CHECK).exists()


@pytest.mark.usefixtures("clock", "claimed")
def test_a_swallowed_error_outside_a_task_is_filed_under_etl(hook: Any) -> None:
    assert tasks._failed("trading_calendar upkeep", KeyError("level")) == "failed"
    assert _bodies(hook)[0]["text"] == "task failed: etl (KeyError) trading_calendar upkeep"


@pytest.mark.usefixtures("clock", "claimed")
def test_a_command_that_failed_on_its_source_is_expected_too(hook: Any) -> None:
    chained = CommandError("SchemaChangedError: FMTQIK fields changed")
    chained.__cause__ = SchemaChangedError("FMTQIK fields changed")
    assert tasks._failed("trading_calendar refresh", chained) == "failed"
    assert len(hook.calls) == 0
    own = CommandError("trading_calendar has no days in 2026-10-01..2026-11-11")
    assert tasks._failed("trading_calendar refresh", own) == "failed"
    assert _bodies(hook)[0]["text"] == "task failed: etl (CommandError) trading_calendar refresh"
