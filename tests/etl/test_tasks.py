from collections.abc import Callable, Iterator
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from celery.schedules import crontab
from django.conf import settings
from django.core.management.base import CommandError

from apps.etl import tasks
from apps.etl.adapters import ADAPTERS
from apps.etl.catchup import CatchupReport
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.quality import CHECKS
from apps.etl.ratelimit import RetryableError, ThrottledError
from apps.etl.trading_calendar import REFRESH_DATASET, REFRESH_SOURCE
from apps.market.calendar import TradingDays
from apps.market.models import IndexDaily, TradingCalendar
from config.celery import app

DAY = date(2026, 10, 28)
TAIPEI = ZoneInfo("Asia/Taipei")
DAILY_JOBS = {(job.source, job.dataset) for jobs in tasks.BATCHES.values() for job in jobs}
SCANS = {("mops", "statement_scan"), ("mops", "revenue_scan")}
MASTER = ("stock_master", "stock_master_daily")
WEEKLY = ("tdcc", "weekly_holding")
SIGNAL = ("datagov", "business_signal")
CALENDAR = (REFRESH_SOURCE, REFRESH_DATASET)

pytestmark = pytest.mark.usefixtures("october_2026_in_db")


@pytest.fixture
def today(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def _set(day: date, at: time = time(23, 0)) -> None:
        moment = datetime.combine(day, at, tzinfo=TAIPEI)
        monkeypatch.setattr("apps.etl.tasks.timezone.localtime", lambda: moment)

    return _set


@pytest.fixture(autouse=True)
def retries_run_in_place() -> Iterator[None]:
    """Eager apply re-runs a retried task at once, unless errors propagate first."""
    before = app.conf.CELERY_TASK_EAGER_PROPAGATES
    app.conf.CELERY_TASK_EAGER_PROPAGATES = False
    yield
    app.conf.CELERY_TASK_EAGER_PROPAGATES = before


@pytest.fixture
def queued(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    sent: list[tuple[str, str, str]] = []
    monkeypatch.setattr(tasks.fetch_day, "delay", lambda *args: sent.append(args))
    return sent


class FakeAdapter:
    """Records runs and raises the queued errors in turn before succeeding."""

    def __init__(self, errors: list[Exception] | None = None) -> None:
        self.errors = list(errors or [])
        self.runs: list[tuple[str, date, date]] = []

    def run(self, dataset: str, start: date, end: date) -> EtlRun:
        self.runs.append((dataset, start, end))
        if self.errors:
            raise self.errors.pop(0)
        return EtlRun(source="fake", dataset=dataset, start_date=start, end_date=end, rows=3)


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[list[tuple[date, str, str]]]:
    calls: list[list[tuple[date, str, str]]] = []
    monkeypatch.setattr(
        "apps.etl.tasks.dispatch",
        lambda rows: calls.append([(row.date, row.check_name, row.severity) for row in rows]),
    )
    return calls


def landed(day: date, jobs: set[tuple[str, str]]) -> None:
    for source, dataset in jobs:
        EtlRun.objects.create(
            source=source, dataset=dataset, start_date=day, end_date=day, status="success"
        )


def refreshed(day: date) -> None:
    """The 15:10 calendar refresh of the session, which covers through the day before."""
    eve = day - timedelta(days=1)
    EtlRun.objects.create(
        source=CALENDAR[0], dataset=CALENDAR[1], start_date=eve, end_date=eve, status="success"
    )


def checked(day: date, *extra: tuple[str, str]) -> None:
    for check, severity in (*((name, "info") for name in CHECKS), *extra):
        DataQualityLog.objects.create(date=day, check_name=check, severity=severity, detail={})


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeAdapter]:
    def _install(*errors: Exception) -> FakeAdapter:
        fake = FakeAdapter(list(errors))
        monkeypatch.setattr("apps.etl.tasks.build_adapter", lambda source, calendar=None: fake)
        return fake

    return _install


def test_the_beat_schedule_covers_the_daily_etl_in_taipei_time() -> None:
    times = {name: entry["schedule"] for name, entry in settings.CELERY_BEAT_SCHEDULE.items()}
    assert times == {
        "etl-prepare-day": crontab(hour=15, minute=10),
        "etl-prices": crontab(hour=15, minute=30),
        "etl-refresh-master": crontab(hour=16, minute=0),
        "etl-afternoon": crontab(hour=16, minute=30),
        "etl-qfii": crontab(hour=18, minute=30),
        "etl-night": crontab(hour=21, minute=30),
        "etl-notices-again": crontab(hour=22, minute=40),
        "etl-statements": crontab(hour=19, minute=0),
        "etl-statements-late": crontab(hour=22, minute=40),
        "etl-revenue": crontab(hour=19, minute=0),
        "etl-revenue-late": crontab(hour=22, minute=40),
        "etl-check-quality": crontab(hour=22, minute=50),
        "etl-us-close": crontab(hour=5, minute=30),
        "etl-weekly-holding": crontab(hour=10, minute=0, day_of_week="sat"),
        "etl-business-signal": crontab(hour=10, minute=0, day_of_month="15"),
    }
    assert settings.CELERY_TIMEZONE == "Asia/Taipei"


def test_every_scheduled_task_and_batch_exists() -> None:
    app.loader.import_default_modules()
    for entry in settings.CELERY_BEAT_SCHEDULE.values():
        assert entry["task"] in app.tasks
        args = entry.get("args", ())
        assert isinstance(args, tuple)
        if entry["task"] == "etl.run_batch":
            assert args[0] in tasks.BATCHES


def test_one_task_at_a_time_and_redelivery_only_after_the_longest_wait() -> None:
    assert settings.CELERY_WORKER_CONCURRENCY == 1
    assert settings.CELERY_TASK_ACKS_LATE
    timeout = settings.CELERY_BROKER_TRANSPORT_OPTIONS["visibility_timeout"]
    assert timeout >= 6 * 3600
    assert timeout > settings.ETL_NOT_READY_COUNTDOWN


def test_the_batches_place_each_dataset_after_its_publication() -> None:
    def keys(batch: str) -> set[str]:
        return {f"{job.source}/{job.dataset}" for job in tasks.BATCHES[batch]}

    assert keys("prices") == {
        "twse/daily_price",
        "tpex/daily_price",
        "twse/index_daily",
        "tpex/index_daily",
        "twse/return_index",
    }
    assert {"twse/sbl", "tpex/sbl", "twse/margin", "tpex/margin"} <= keys("night")
    assert {"twse/dividend_event", "tpex/dividend_event"} <= keys("night")
    assert "dividend_event" in tasks.GATE_DATASETS
    assert keys("qfii") == {"tpex/foreign_holding", "tpex/shares_outstanding"}
    assert keys("notices") == {
        "twse/attention",
        "tpex/attention",
        "twse/disposition",
        "tpex/disposition",
    }
    trade = [job for job in tasks.BATCHES["night"] if job.dataset == "day_trade"]
    assert {job.source for job in trade} == {"twse", "tpex"}
    assert all(job.sessions_back == (0, 1, 2) for job in trade)
    daily = {f"{job.source}/{job.dataset}" for jobs in tasks.BATCHES.values() for job in jobs}
    for source in ("twse", "tpex"):
        assert {key for key in daily if key.startswith(source)} == {
            f"{source}/{dataset}" for dataset in ADAPTERS[source].datasets
        }


def test_a_batch_queues_one_task_per_dataset_and_session(
    today: Callable[..., None], queued: list[tuple[str, str, str]]
) -> None:
    today(DAY)
    tasks.run_batch("night")
    assert ("twse", "day_trade", "2026-10-28") in queued
    assert ("twse", "day_trade", "2026-10-27") in queued
    assert ("twse", "day_trade", "2026-10-22") in queued
    assert ("tpex", "margin", "2026-10-28") in queued
    assert len(queued) == len(set(queued))


def test_nothing_runs_on_a_day_without_trading(
    today: Callable[..., None], queued: list[tuple[str, str, str]]
) -> None:
    today(date(2026, 10, 25))
    assert tasks.run_batch("prices") == []
    assert queued == []


def test_a_batch_fired_late_on_waking_before_its_slot_is_skipped(
    today: Callable[..., None],
    queued: list[tuple[str, str, str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    today(DAY, time(8, 0))
    assert tasks.run_batch("prices") == []
    assert tasks.check_quality() == 0
    assert tasks.refresh_master.apply().get() == "closed"
    assert queued == []
    skipped = [row for row in caplog.records if "etl skipped" in row.getMessage()]
    assert [row.levelname for row in skipped] == ["WARNING"] * 3
    assert "PRICES" in skipped[0].getMessage()


def test_a_dataset_landing_after_a_failed_check_runs_the_checks_again(
    adapter: Callable[..., FakeAdapter], sent: list[list[tuple[date, str, str]]]
) -> None:
    adapter()
    DataQualityLog.objects.create(
        date=DAY, check_name="missing_values", severity="error", detail={}
    )
    tasks.fetch_day.apply(args=("tpex", "margin", "2026-10-28")).get()
    assert not DataQualityLog.objects.filter(date=DAY, severity="error").exists()
    assert DataQualityLog.objects.filter(date=DAY, check_name="big_move").exists()
    DataQualityLog.objects.filter(date=DAY, check_name="big_move").update(detail={"again": 1})
    tasks.fetch_day.apply(args=("tpex", "margin", "2026-10-28")).get()
    assert DataQualityLog.objects.get(date=DAY, check_name="big_move").detail == {"again": 1}


def test_an_alert_still_standing_after_a_recheck_is_not_sent_again(
    adapter: Callable[..., FakeAdapter],
    sent: list[list[tuple[date, str, str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter()
    standing = [("missing_values", "error")]

    def checks(day: date, calendar: TradingDays) -> list[DataQualityLog]:
        DataQualityLog.objects.filter(date=day, check_name__in=CHECKS).delete()
        return [
            DataQualityLog.objects.create(date=day, check_name=name, severity=level, detail={})
            for name, level in standing
        ]

    monkeypatch.setattr("apps.etl.tasks.run_checks", checks)
    checks(DAY, TradingDays.from_db())
    tasks.fetch_day.apply(args=("twse", "day_trade", "2026-10-28")).get()
    assert sent == []
    standing.append(("row_count", "warning"))
    tasks.fetch_day.apply(args=("twse", "day_trade", "2026-10-28")).get()
    assert sent == [[(DAY, "row_count", "warning")]]


def test_a_gate_dataset_landing_late_refreshes_the_landed_row(
    adapter: Callable[..., FakeAdapter], sent: list[list[tuple[date, str, str]]]
) -> None:
    adapter()
    checked(DAY, ("etl_runs", "error"))
    landed(DAY, {job for job in DAILY_JOBS if job[1] in tasks.GATE_DATASETS})
    tasks.fetch_day.apply(args=("tpex", "margin", "2026-10-28")).get()
    row = DataQualityLog.objects.get(date=DAY, check_name="etl_runs")
    assert (row.severity, row.detail["blocking"]) == ("warning", [])
    assert sent == [[(DAY, "etl_runs", "warning")]]


def test_a_dataset_runs_for_its_day(adapter: Callable[..., FakeAdapter]) -> None:
    fake = adapter()
    result = tasks.fetch_day.apply(args=("twse", "daily_price", "2026-10-28")).get()
    assert fake.runs == [("daily_price", DAY, DAY)]
    assert result == "success: 3 rows"


@pytest.mark.parametrize(
    ("error", "outcome"),
    [(SourceNotReadyError("not yet"), "not ready"), (RetryableError("timed out"), "unreachable")],
)
def test_a_day_not_published_or_unreachable_is_retried_until_the_limit(
    adapter: Callable[..., FakeAdapter],
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    outcome: str,
) -> None:
    limit = settings.ETL_NOT_READY_RETRIES
    fake = adapter(*(error for _ in range(limit + 1)))
    result = tasks.fetch_day.apply(args=("twse", "sbl", "2026-10-28")).get()
    assert len(fake.runs) == limit + 1
    assert result == outcome
    assert any("gave up" in row.getMessage() for row in caplog.records)


def test_a_day_published_on_a_retry_is_loaded(adapter: Callable[..., FakeAdapter]) -> None:
    fake = adapter(SourceNotReadyError("not yet"), RetryableError("connection reset"))
    result = tasks.fetch_day.apply(args=("twse", "sbl", "2026-10-28")).get()
    assert (len(fake.runs), result) == (3, "success: 3 rows")


@pytest.mark.parametrize(
    "error", [SchemaChangedError("layout changed"), ThrottledError("still throttled")]
)
def test_any_other_error_fails_without_retrying(
    adapter: Callable[..., FakeAdapter], caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    fake = adapter(error)
    result = tasks.fetch_day.apply(args=("twse", "sbl", "2026-10-28")).get()
    assert (len(fake.runs), result) == (1, "failed")
    assert [row.levelname for row in caplog.records if "etl failed" in row.getMessage()] == [
        "ERROR"
    ]


@pytest.fixture
def catch_ups(monkeypatch: pytest.MonkeyPatch) -> list[tuple[date, date, TradingDays]]:
    calls: list[tuple[date, date, TradingDays]] = []

    def fake_catch_up(since: date, until: date, calendar: TradingDays) -> Any:
        calls.append((since, until, calendar))
        return CatchupReport({}, {}, {}, {}, [])

    monkeypatch.setattr("apps.etl.tasks.catch_up", fake_catch_up)
    return calls


def test_the_day_starts_with_the_calendar_then_missed_days(
    today: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    catch_ups: list[tuple[date, date, TradingDays]],
) -> None:
    calls: list[str] = []

    def refresh(name: str, *args: Any, **kwargs: Any) -> None:
        calls.append(name)
        raise CommandError("SchemaChangedError: FMTQIK fields changed")

    monkeypatch.setattr("apps.etl.tasks.call_command", refresh)
    today(DAY, time(15, 10))
    tasks.prepare_day()
    assert calls == ["refresh_calendar"]
    assert [call[:2] for call in catch_ups] == [(date(2026, 10, 14), date(2026, 10, 27))]
    audited = set(DataQualityLog.objects.values_list("date", flat=True))
    assert audited == {date(2026, 10, 27)}


def test_missed_days_are_fetched_with_the_corrected_calendar(
    today: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    catch_ups: list[tuple[date, date, TradingDays]],
) -> None:
    def close_a_day(name: str, *args: Any, **kwargs: Any) -> None:
        TradingCalendar.objects.filter(date=date(2026, 10, 27)).update(is_trading_day=False)

    monkeypatch.setattr("apps.etl.tasks.call_command", close_a_day)
    today(DAY, time(15, 10))
    tasks.prepare_day()
    [(_, until, calendar)] = catch_ups
    assert until == date(2026, 10, 22)
    assert not calendar.is_trading_day(date(2026, 10, 27))


@pytest.mark.usefixtures("catch_ups")
def test_a_session_already_audited_is_left_alone(
    today: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    sent: list[list[tuple[date, str, str]]],
) -> None:
    monkeypatch.setattr("apps.etl.tasks.call_command", lambda name, **kwargs: None)
    checked(date(2026, 10, 27), ("etl_runs", "info"))
    before = set(DataQualityLog.objects.values_list("id", flat=True))
    today(DAY, time(15, 10))
    tasks.prepare_day()
    assert set(DataQualityLog.objects.values_list("id", flat=True)) == before
    assert sent == []


@pytest.mark.usefixtures("catch_ups")
def test_sessions_missed_since_the_schedule_began_are_audited(
    today: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    sent: list[list[tuple[date, str, str]]],
) -> None:
    monkeypatch.setattr("apps.etl.tasks.call_command", lambda name, **kwargs: None)
    began = date(2026, 10, 20)
    checked(began, ("etl_runs", "error"))
    landed(began, DAILY_JOBS)
    today(DAY, time(15, 10))
    tasks.prepare_day()
    audited = DataQualityLog.objects.filter(check_name="etl_runs")
    since_start = TradingDays.from_db().between(began, date(2026, 10, 27))
    assert len(since_start) > 2
    assert set(audited.values_list("date", flat=True)) == set(since_start)
    assert audited.get(date=began).severity == "info"
    assert {call[0][0] for call in sent} == set(since_start[1:])
    listed = [key for row in audited for key in row.detail["missing"]]
    assert listed
    owned = {
        f"{source}/{dataset}" for source, dataset in (*SCANS, MASTER, WEEKLY, SIGNAL, CALENDAR)
    }
    assert not [key for key in listed if key in owned]


def test_a_day_without_trading_is_not_prepared(
    today: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("apps.etl.tasks.call_command", lambda name, **kwargs: calls.append(name))
    today(date(2026, 10, 25), time(15, 10))
    assert tasks.prepare_day() == []
    assert calls == []
    assert not EtlRun.objects.exists()


def test_the_master_refresh_waits_for_the_days_quotes(
    today: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[date] = []

    def refresh(day: date, calendar: TradingDays) -> EtlRun:
        attempts.append(day)
        if len(attempts) < 3:
            raise SourceNotReadyError("quotes not loaded")
        return EtlRun(start_date=day, end_date=day, checkpoint={"listed": 1})

    monkeypatch.setattr("apps.etl.tasks.refresh_stock_master", refresh)
    today(DAY)
    assert tasks.refresh_master.apply().get() == "success"
    assert attempts == [DAY, DAY, DAY]


def test_quality_is_checked_on_trading_days_only(
    today: Callable[..., None], sent: list[list[tuple[date, str, str]]]
) -> None:
    today(date(2026, 10, 25))
    tasks.check_quality()
    assert not DataQualityLog.objects.exists()
    today(DAY)
    tasks.check_quality()
    assert set(DataQualityLog.objects.values_list("date", flat=True)) == {DAY}
    assert len(sent) == 1
    row = DataQualityLog.objects.get(check_name="etl_runs")
    assert row.severity == "error"
    assert "tpex/full_delivery" in row.detail["missing"]
    assert "twse/daily_price" in row.detail["blocking"]
    assert "tpex/full_delivery" not in row.detail["blocking"]


@pytest.mark.parametrize(
    ("absent", "severity"),
    [(set(), "info"), ({("tpex", "full_delivery"), ("twse", "day_trade")}, "warning")],
)
def test_only_a_missing_gate_dataset_makes_the_landed_row_an_error(
    today: Callable[..., None],
    sent: list[list[tuple[date, str, str]]],
    absent: set[tuple[str, str]],
    severity: str,
) -> None:
    landed(DAY, (DAILY_JOBS | SCANS | {MASTER}) - absent)
    refreshed(DAY)
    today(DAY)
    tasks.check_quality()
    row = DataQualityLog.objects.get(check_name="etl_runs")
    assert row.severity == severity
    assert row.detail == {"missing": sorted(f"{s}/{d}" for s, d in absent), "blocking": []}


def test_a_scan_due_without_a_successful_run_warns(
    today: Callable[..., None],
    sent: list[list[tuple[date, str, str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    landed(DAY, DAILY_JOBS | {MASTER})
    refreshed(DAY)
    EtlRun.objects.create(
        source="mops", dataset="revenue_scan", start_date=DAY, end_date=DAY, status="failed"
    )
    today(DAY)
    tasks.check_quality()
    row = DataQualityLog.objects.get(check_name="etl_runs")
    assert row.severity == "warning"
    assert row.detail == {"missing": ["mops/revenue_scan", "mops/statement_scan"], "blocking": []}
    months = tasks.open_months
    monkeypatch.setattr("apps.etl.tasks.open_months", lambda day, days_after: [])
    tasks.check_quality()
    assert DataQualityLog.objects.get(check_name="etl_runs").detail["missing"] == [
        "mops/statement_scan"
    ]
    monkeypatch.setattr("apps.etl.tasks.open_months", months)
    monkeypatch.setattr("apps.etl.tasks.open_periods", lambda day, days_after: [])
    tasks.check_quality()
    assert DataQualityLog.objects.get(check_name="etl_runs").detail["missing"] == [
        "mops/revenue_scan"
    ]


def missing_on(day: date, today: Callable[..., None]) -> list[str]:
    today(day)
    tasks.check_quality()
    row = DataQualityLog.objects.get(date=day, check_name="etl_runs")
    assert row.detail["blocking"] == []
    missing: list[str] = row.detail["missing"]
    return missing


@pytest.mark.usefixtures("sent")
def test_the_evening_check_lists_a_master_refresh_that_did_not_run(
    today: Callable[..., None],
) -> None:
    landed(DAY, DAILY_JOBS | SCANS)
    refreshed(DAY)
    EtlRun.objects.create(
        source=MASTER[0], dataset=MASTER[1], start_date=DAY, end_date=DAY, status="failed"
    )
    assert missing_on(DAY, today) == ["stock_master/stock_master_daily"]
    assert DataQualityLog.objects.get(check_name="etl_runs").severity == "warning"
    landed(DAY, {MASTER})
    assert missing_on(DAY, today) == []


@pytest.mark.usefixtures("sent")
def test_the_first_session_after_a_saturday_owns_that_weeks_holding_file(
    today: Callable[..., None],
) -> None:
    after, later = date(2026, 10, 27), date(2026, 10, 28)
    for day in (after, later):
        landed(day, DAILY_JOBS | SCANS | {MASTER})
        refreshed(day)
    EtlRun.objects.create(
        source=WEEKLY[0], dataset=WEEKLY[1], start_date=after, end_date=after, status="success"
    )
    assert missing_on(after, today) == ["tdcc/weekly_holding"]
    assert missing_on(later, today) == []
    landed(date(2026, 10, 22), {WEEKLY})
    assert missing_on(after, today) == []


@pytest.mark.usefixtures("sent")
def test_the_session_on_the_15th_owns_the_monthly_business_signal(
    today: Callable[..., None],
) -> None:
    due, later = date(2026, 10, 15), date(2026, 10, 16)
    for day in (due, later):
        landed(day, DAILY_JOBS | SCANS | {MASTER})
        refreshed(day)
    assert missing_on(due, today) == ["datagov/business_signal"]
    assert missing_on(later, today) == []
    landed(due, {SIGNAL})
    assert missing_on(due, today) == []


def test_the_monthly_due_day_follows_the_beat_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tasks.signal_days() == {15}
    beat = {**settings.CELERY_BEAT_SCHEDULE}
    beat["etl-business-signal"] = {"schedule": crontab(hour=10, minute=0, day_of_month="1,20")}
    monkeypatch.setattr(settings, "CELERY_BEAT_SCHEDULE", beat)
    assert tasks.signal_days() == {1, 20}
    calendar = TradingDays.from_db()
    signal = tasks.Due(*SIGNAL, date(2026, 10, 20))
    assert signal in tasks._periodic_due(date(2026, 10, 20), calendar)
    assert not [
        due for due in tasks._periodic_due(DAY.replace(day=15), calendar) if due[:2] == SIGNAL
    ]


@pytest.mark.usefixtures("sent")
def test_a_due_day_off_the_calendar_belongs_to_the_next_session(
    today: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    beat = {**settings.CELERY_BEAT_SCHEDULE}
    beat["etl-business-signal"] = {"schedule": crontab(hour=10, minute=0, day_of_month="10")}
    monkeypatch.setattr(settings, "CELERY_BEAT_SCHEDULE", beat)
    session = date(2026, 10, 12)
    landed(session, DAILY_JOBS | SCANS | {MASTER, WEEKLY})
    refreshed(session)
    assert missing_on(session, today) == ["datagov/business_signal", "tdcc/weekly_holding"]
    landed(date(2026, 10, 10), {SIGNAL})
    landed(date(2026, 10, 8), {WEEKLY})
    assert missing_on(session, today) == []


@pytest.mark.usefixtures("sent")
def test_the_evening_check_lists_a_calendar_refresh_that_did_not_run(
    today: Callable[..., None],
) -> None:
    landed(DAY, DAILY_JOBS | SCANS | {MASTER})
    assert missing_on(DAY, today) == ["trading_calendar/trading_calendar_refresh"]
    EtlRun.objects.create(
        source=CALENDAR[0],
        dataset=CALENDAR[1],
        start_date=date(2026, 10, 1),
        end_date=DAY - timedelta(days=1),
        status="success",
    )
    assert missing_on(DAY, today) == []


@pytest.mark.usefixtures("catch_ups", "sent")
def test_a_listed_periodic_miss_stays_listed_until_it_lands(
    today: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.tasks.call_command", lambda name, **kwargs: None)
    landed(DAY, DAILY_JOBS | SCANS)
    refreshed(DAY)
    assert missing_on(DAY, today) == ["stock_master/stock_master_daily"]
    next_day = date(2026, 10, 29)
    today(next_day, time(15, 10))
    tasks.prepare_day()
    row = DataQualityLog.objects.get(date=DAY, check_name="etl_runs")
    assert (row.severity, row.detail["missing"]) == ("warning", ["stock_master/stock_master_daily"])
    landed(DAY, {MASTER})
    tasks.prepare_day()
    assert DataQualityLog.objects.get(date=DAY, check_name="etl_runs").severity == "info"


def test_the_us_close_loads_the_last_session_and_logs_stale_series(
    today: Callable[..., None], adapter: Callable[..., FakeAdapter]
) -> None:
    fake = adapter()
    IndexDaily.objects.create(index_id="SOX", date=DAY, close=1)
    IndexDaily.objects.create(index_id="USDTWD", date=date(2026, 10, 22), close=1)
    today(date(2026, 10, 29), time(5, 30))
    tasks.us_close()
    assert sorted(fake.runs) == [(name, DAY, DAY) for name in ("sox", "tsm_adr", "usdtwd")]
    row = DataQualityLog.objects.get(check_name="us_close")
    assert (row.date, row.severity) == (DAY, "warning")
    assert row.detail == {"stale": {"TSM_ADR": "", "USDTWD": "2026-10-22"}}


@pytest.mark.usefixtures("adapter")
def test_usdtwd_a_session_behind_is_fresh_at_dawn(today: Callable[..., None]) -> None:
    IndexDaily.objects.create(index_id="SOX", date=DAY, close=1)
    IndexDaily.objects.create(index_id="TSM_ADR", date=DAY, close=1)
    IndexDaily.objects.create(index_id="USDTWD", date=date(2026, 10, 27), close=1)
    today(date(2026, 10, 29), time(5, 30))
    tasks.us_close()
    row = DataQualityLog.objects.get(check_name="us_close")
    assert (row.severity, row.detail) == ("info", {"stale": {}})


def test_the_us_close_fired_before_its_slot_waits_for_it(
    today: Callable[..., None], adapter: Callable[..., FakeAdapter]
) -> None:
    fake = adapter()
    today(date(2026, 10, 29), time(3, 0))
    assert tasks.us_close() == "early"
    assert fake.runs == []
    assert not DataQualityLog.objects.exists()


def test_the_us_close_of_a_session_runs_once(
    today: Callable[..., None], adapter: Callable[..., FakeAdapter]
) -> None:
    fake = adapter()
    today(date(2026, 10, 31), time(5, 30))
    tasks.us_close()
    today(date(2026, 11, 1), time(5, 30))
    assert tasks.us_close() == "done"
    assert len(fake.runs) == 3


def test_one_series_failing_still_loads_the_others_and_logs(
    today: Callable[..., None],
    adapter: Callable[..., FakeAdapter],
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = adapter(SchemaChangedError("layout changed"))
    today(date(2026, 10, 29), time(5, 30))
    assert tasks.us_close() == "warning"
    assert len(fake.runs) == 3
    assert any("etl failed yahoo/" in row.getMessage() for row in caplog.records)


@pytest.mark.parametrize("run_on", [date(2026, 10, 31), date(2026, 11, 2)])
def test_the_weekly_holding_asks_for_the_last_ended_weeks_last_session(
    today: Callable[..., None], adapter: Callable[..., FakeAdapter], run_on: date
) -> None:
    fake = adapter()
    today(run_on, time(10, 0))
    tasks.weekly_holding.apply().get()
    assert fake.runs == [("weekly_holding", date(2026, 10, 30), date(2026, 10, 30))]


def test_a_week_without_sessions_has_no_holding_to_fetch(
    today: Callable[..., None], adapter: Callable[..., FakeAdapter]
) -> None:
    fake = adapter()
    TradingCalendar.objects.filter(date__range=(date(2026, 10, 5), date(2026, 10, 10))).update(
        is_trading_day=False
    )
    today(date(2026, 10, 12))
    assert tasks.weekly_holding.apply().get() == "no session this week"
    assert fake.runs == []


@pytest.mark.usefixtures("catch_ups", "sent")
def test_a_later_master_refresh_clears_a_missed_one_but_a_later_week_does_not(
    today: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.tasks.call_command", lambda name, **kwargs: None)
    session = date(2026, 10, 27)
    landed(session, DAILY_JOBS | SCANS)
    refreshed(session)
    assert missing_on(session, today) == ["stock_master/stock_master_daily", "tdcc/weekly_holding"]
    landed(date(2026, 10, 28), {MASTER})
    landed(date(2026, 10, 30), {WEEKLY})
    today(date(2026, 10, 29), time(15, 10))
    tasks.prepare_day()
    row = DataQualityLog.objects.get(date=session, check_name="etl_runs")
    assert (row.severity, row.detail["missing"]) == ("warning", ["tdcc/weekly_holding"])


def test_a_week_without_sessions_owes_no_holding_file() -> None:
    TradingCalendar.objects.filter(date__range=(date(2026, 10, 19), date(2026, 10, 22))).update(
        is_trading_day=False
    )
    calendar = TradingDays.from_db()
    weekly = [due for due in tasks._periodic_due(date(2026, 10, 27), calendar) if due[:2] == WEEKLY]
    assert weekly == [tasks.Due(*WEEKLY, date(2026, 10, 16))]


def test_the_first_session_of_the_calendar_owes_only_the_daily_refreshes() -> None:
    first = date(2026, 10, 1)
    due = tasks._periodic_due(first, TradingDays.from_db())
    assert {item[:2] for item in due} == {MASTER, CALENDAR}


@pytest.mark.parametrize(
    ("ran", "recorded"),
    [
        (date(2026, 10, 15), date(2026, 10, 15)),
        (date(2026, 10, 18), date(2026, 10, 15)),
        (date(2026, 10, 3), date(2026, 10, 3)),
    ],
)
def test_the_business_signal_is_recorded_on_its_due_day_even_when_late(
    today: Callable[..., None], adapter: Callable[..., FakeAdapter], ran: date, recorded: date
) -> None:
    fake = adapter()
    today(ran)
    tasks.business_signal.apply().get()
    assert fake.runs == [("business_signal", recorded, recorded)]


def test_the_last_statement_scan_is_where_filings_stop_counting_for_the_session() -> None:
    assert settings.ETL_AT["STATEMENTS_LATE"] == settings.FILING_VISIBLE_BEFORE
    assert settings.ETL_AT["STATEMENTS_LATE"] < settings.ETL_AT["CHECK_QUALITY"]
    entries = {
        entry["args"]
        for entry in settings.CELERY_BEAT_SCHEDULE.values()
        if entry["task"] == "etl.statement_scan"
    }
    assert entries == {("STATEMENTS",), ("STATEMENTS_LATE",)}


def test_the_statement_scan_runs_today_while_a_quarter_is_open(
    today: Callable[..., None], adapter: Callable[..., FakeAdapter]
) -> None:
    fake = adapter()
    today(DAY, time(19, 0))
    assert tasks.statement_scan.apply(args=("STATEMENTS",)).get() == "success: 3 rows"
    assert fake.runs == [("statement_scan", DAY, DAY)]


def test_the_statement_scan_skips_closed_days_early_slots_and_quiet_seasons(
    today: Callable[..., None],
    adapter: Callable[..., FakeAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = adapter()
    today(date(2026, 10, 25), time(19, 0))
    assert tasks.statement_scan.apply(args=("STATEMENTS",)).get() == "closed"
    today(DAY, time(18, 59))
    assert tasks.statement_scan.apply(args=("STATEMENTS",)).get() == "closed"
    monkeypatch.setattr("apps.etl.tasks.open_periods", lambda day, days_after: [])
    today(DAY, time(19, 0))
    assert tasks.statement_scan.apply(args=("STATEMENTS",)).get() == "no period open"
    assert fake.runs == []


def test_an_unreachable_statement_scan_is_retried(
    today: Callable[..., None], adapter: Callable[..., FakeAdapter]
) -> None:
    fake = adapter(RetryableError("connection reset"))
    today(DAY, time(22, 40))
    assert tasks.statement_scan.apply(args=("STATEMENTS_LATE",)).get() == "success: 3 rows"
    assert len(fake.runs) == 2


def test_the_last_revenue_scan_runs_with_the_last_statement_scan() -> None:
    assert settings.ETL_AT["REVENUE_LATE"] == settings.FILING_VISIBLE_BEFORE
    entries = {
        entry["args"]
        for entry in settings.CELERY_BEAT_SCHEDULE.values()
        if entry["task"] == "etl.revenue_scan"
    }
    assert entries == {("REVENUE",), ("REVENUE_LATE",)}


def test_the_revenue_scan_runs_today_while_a_month_is_open(
    today: Callable[..., None], adapter: Callable[..., FakeAdapter]
) -> None:
    fake = adapter()
    today(DAY, time(19, 0))
    assert tasks.revenue_scan.apply(args=("REVENUE",)).get() == "success: 3 rows"
    assert fake.runs == [("revenue_scan", DAY, DAY)]


def test_the_revenue_scan_skips_closed_days_early_slots_and_quiet_days(
    today: Callable[..., None],
    adapter: Callable[..., FakeAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = adapter()
    today(date(2026, 10, 25), time(19, 0))
    assert tasks.revenue_scan.apply(args=("REVENUE",)).get() == "closed"
    today(DAY, time(18, 59))
    assert tasks.revenue_scan.apply(args=("REVENUE",)).get() == "closed"
    monkeypatch.setattr("apps.etl.tasks.open_months", lambda day, days_after: [])
    today(DAY, time(19, 0))
    assert tasks.revenue_scan.apply(args=("REVENUE",)).get() == "no month open"
    assert fake.runs == []


def test_an_unreachable_revenue_scan_is_retried(
    today: Callable[..., None], adapter: Callable[..., FakeAdapter]
) -> None:
    fake = adapter(RetryableError("connection reset"))
    today(DAY, time(22, 40))
    assert tasks.revenue_scan.apply(args=("REVENUE_LATE",)).get() == "success: 3 rows"
    assert len(fake.runs) == 2


def test_research_waits_for_the_gate_datasets_and_the_quality_gate() -> None:
    assert not tasks.ready_for_research(DAY)
    gate = {job for job in DAILY_JOBS if job[1] in tasks.GATE_DATASETS}
    assert {dataset for _, dataset in DAILY_JOBS - gate} == {
        "attention",
        "disposition",
        "full_delivery",
        "shares_outstanding",
        "day_trade",
        "return_index",
    }
    landed(DAY, gate)
    assert tasks.etl_complete(DAY)
    assert not tasks.ready_for_research(DAY)
    checked(DAY)
    assert tasks.ready_for_research(DAY)
    EtlRun.objects.filter(dataset="margin", source="tpex").update(status="failed")
    assert not tasks.etl_complete(DAY)
    assert not tasks.ready_for_research(DAY)
