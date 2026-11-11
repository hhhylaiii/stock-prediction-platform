"""Celery tasks behind the 23:00 and 06:00 research schedule (SPEC §5 每日增量)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, time, timedelta
from io import StringIO
from time import monotonic
from typing import Any

from celery import Task, chain, shared_task
from django.conf import settings
from django.core.management import call_command
from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from apps.etl.alerts import dispatch
from apps.etl.models import DataQualityLog
from apps.etl.tasks import US_CLOSE_CHECK, ready_for_research
from apps.market.adjust import ADJ_CHECK
from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.research.feature_coverage import CHECK as COVERAGE_CHECK
from apps.research.feature_coverage import CoverageError, coverage_summary, run_coverage
from apps.research.labels import HORIZONS
from apps.research.lock import ResearchBusyError, research_lock
from apps.research.models import HoldoutScope, LabelDaily, ModelRun
from apps.research.monitor import run_monitor
from apps.research.predict import predict
from apps.research.production import DESIGNS, build_run, shadow_key
from apps.research.promotion import PROMOTION_CHECK, first_day, promote
from apps.research.resources import free_percent, peak_rss_mb
from apps.research.runs import PRODUCTION

logger = logging.getLogger(__name__)
EVENING_CHECK = "research_evening"
MORNING_CHECK = "research_morning"
RETRAIN_CHECK = "research_retrain"
PROMOTE_CHECK = "research_promote"
REBUILD_CHECK = "research_rebuild"
RECENT_SESSIONS = 60
COVERED = f"covered by the chain's {RECENT_SESSIONS}-session features and labels"
NOTE_WINDOW = 90
CONCURRENT = "the research schedule runs on a single worker process"
NOT_FROZEN = "the design is not frozen: DESIGNS names no research_run_key"
ALREADY_BUILT = "already built this month"
MISSED = "missed"
MISSED_LOOKBACK = 5
RETRAIN_WINDOW = 5
SKIPPED = (
    f"no morning chain finished before the cutoff in the month's first {RETRAIN_WINDOW} sessions"
)
GATE_SHUT = "research gate shut"
US_CLOSE_MISSING = "missing"
EVENING_STEPS = ("recent_features", "labels")
MORNING_STEPS = ("features", "coverage", "predict", "monitor")
Severity = DataQualityLog.Severity
Done = dict[str, str]


def _command(name: str, **options: Any) -> str:
    out = StringIO()
    call_command(name, stdout=out, **options)
    return out.getvalue().strip()


def _features(day: date) -> str:
    return _command("build_features", date=day)


def _recent_features(day: date) -> str:
    return _command("build_features", end=day, recent=RECENT_SESSIONS)


def _labels(day: date) -> str:
    """Labels far enough back to reach every one a change in the feature window moved."""
    return _command("build_labels", end=day, recent=RECENT_SESSIONS + max(HORIZONS) + 1)


def _coverage(day: date) -> str:
    """The day's feature coverage row; a check that cannot run warns and the chain goes on.

    An earlier error row of the day stands, so a failed recheck never reopens its predictions.
    """
    try:
        row = run_coverage(day, TradingDays.from_db(), settings.FEATURE_PARQUET_DIR)
    except Exception as error:
        logger.error("research coverage %s failed: %s", day, error)
        shown = str(error) if isinstance(error, CoverageError) else type(error).__name__
        stands = DataQualityLog.objects.filter(
            date=day, check_name=COVERAGE_CHECK, severity=Severity.ERROR
        ).exists()
        if stands:
            return f"coverage check failed ({shown}); the day's earlier error stands"
        _record(day, COVERAGE_CHECK, Severity.WARNING, {"error": shown})
        return f"coverage check failed, see {COVERAGE_CHECK}"
    return coverage_summary(row)


def _predict(day: date) -> str:
    result = predict(day, settings.FEATURE_PARQUET_DIR)
    written = [f"{run_id}: {count} rows" for run_id, count in result.written.items()]
    return "; ".join([*written, *(f"no serving run for {slot}" for slot in result.missing)])


def _monitor(day: date) -> str:
    result = run_monitor(day, TradingDays.from_db())
    degraded = ", ".join(result.degraded) or "none"
    return f"{result.scored} model_monitor rows, degraded {degraded}"


def retrain_key(scope: str, day: date) -> str:
    return f"{scope}-{day:%Y-%m}"


def _month(check: str, day: date) -> Any:
    return DataQualityLog.objects.filter(
        check_name=check, date__year=day.year, date__month=day.month
    )


def _tried(keys: list[str]) -> set[str]:
    """Run keys a promotion judged, or that already failed once in the promote step."""
    judged = DataQualityLog.objects.filter(check_name=PROMOTION_CHECK, detail__run_key__in=keys)
    errored = DataQualityLog.objects.filter(
        check_name=PROMOTE_CHECK, detail__failed__has_any_keys=keys
    )
    return {row.detail["run_key"] for row in judged} | {
        key for row in errored for key in row.detail["failed"]
    }


def _pending(day: date) -> list[str]:
    """The run keys this month's retrain built that no promotion has tried yet."""
    row = _month(RETRAIN_CHECK, day).filter(date__lte=day).order_by("-date", "-pk").first()
    built = {} if row is None else row.detail.get("built", {})
    keys = [built[scope] for scope in HoldoutScope.values if scope in built]
    if not keys:
        return []
    tried = _tried(keys)
    return [key for key in keys if key not in tried]


def _promote(day: date) -> str:
    """Promote each untried build from day, or its first day if later; errors make one warning."""
    done: list[str] = []
    failed: dict[str, str] = {}
    for run_key in _pending(day):
        try:
            result = promote(run_key, max(day, first_day(run_key)), settings.FEATURE_PARQUET_DIR)
        except Exception as error:
            logger.error("research promote %s %s failed: %s", run_key, day, error)
            failed[run_key] = f"{type(error).__name__}: {error}"
            continue
        done.append(f"{run_key}: {'promoted' if result.promoted else 'kept the serving runs'}")
    if failed:
        _record(day, PROMOTE_CHECK, Severity.WARNING, {"failed": failed, "done": done})
    return "; ".join([*done, *(f"{key}: failed, see {PROMOTE_CHECK}" for key in failed)])


def _taken(since: date) -> dict[str, list[Any]]:
    """Every content a rebuild took each adj_factor note in at, by the note's pk."""
    rows = DataQualityLog.objects.filter(
        check_name=REBUILD_CHECK, date__gte=since - timedelta(days=NOTE_WINDOW)
    )
    taken: dict[str, list[Any]] = {}
    for row in rows:
        for pk, detail in row.detail.get("notes", {}).items():
            taken.setdefault(pk, []).append(detail)
    return taken


def _open_notes() -> list[DataQualityLog]:
    """The recent adj_factor_changed rows no rebuild took in as they read now."""
    since = timezone.localtime().date() - timedelta(days=NOTE_WINDOW)
    taken = _taken(since)
    notes = DataQualityLog.objects.filter(check_name=ADJ_CHECK, date__gte=since).order_by("pk")
    return [note for note in notes if note.detail not in taken.get(str(note.pk), [])]


def _earliest(note: DataQualityLog) -> date:
    stocks = note.detail.get("stocks", {})
    return date.fromisoformat(note.detail.get("earliest") or min(stocks.values()))


def _stocks(notes: list[DataQualityLog]) -> int:
    named = {stock for note in notes for stock in note.detail.get("stocks", {})}
    return len(named) + sum(int(note.detail.get("count", 0)) for note in notes)


def _append(day: date, check: str, severity: str, detail: dict[str, Any]) -> None:
    row = DataQualityLog.objects.create(
        date=day, check_name=check, severity=severity, detail=detail
    )
    if severity != Severity.INFO:
        dispatch([row])


def _rebuild_range(calendar: TradingDays, earliest: date, day: date) -> str:
    """Labels from h_max+1 sessions before earliest and features up to the day before day."""
    start = calendar.shift(earliest, -(max(HORIZONS) + 1))
    done = [_command("build_labels", start=start, end=day)]
    if earliest < day:
        done.append(_command("build_features", start=earliest, end=calendar.shift(day, -1)))
    return "; ".join(done)


def _summary(notes: list[DataQualityLog], handled: str) -> dict[str, Any]:
    earliest = min(_earliest(note) for note in notes)
    taken = {str(note.pk): note.detail for note in notes}
    return {
        "earliest": earliest.isoformat(),
        "stocks": _stocks(notes),
        "notes": taken,
        "limit_days": RECENT_SESSIONS,
        "handled": handled,
    }


def _rebuild_notes(day: date, build: bool) -> str:
    """Notes past the feature window go to a person; the rest are rebuilt or found covered."""
    notes = [note for note in _open_notes() if _earliest(note) <= day]
    if not notes:
        return f"no adj_factor change up to {day}"
    calendar = TradingDays.from_db()
    cutoff = calendar.shift(day, 1 - RECENT_SESSIONS)
    old = [note for note in notes if _earliest(note) < cutoff]
    recent = [note for note in notes if _earliest(note) >= cutoff]
    done = []
    if old:
        limited = _summary(old, "by_hand")
        _append(day, REBUILD_CHECK, Severity.WARNING, limited)
        outside = f"outside the last {RECENT_SESSIONS} sessions"
        done.append(f"adj_factor moved from {limited['earliest']}, {outside}")
    if recent:
        detail = _summary(recent, "rebuilt" if build else "covered")
        earliest = date.fromisoformat(detail["earliest"])
        built = _rebuild_range(calendar, earliest, day) if build else COVERED
        _append(day, REBUILD_CHECK, Severity.INFO, detail)
        done.append(built)
    return "; ".join(done)


def _rebuild(day: date, build: bool = True) -> str:
    """Take in the adj_factor changes up to day; a failure warns and leaves them open."""
    try:
        return _rebuild_notes(day, build)
    except Exception as error:
        logger.error("research rebuild %s failed: %s", day, error)
        failed = {
            "error": f"{type(error).__name__}: {error}",
            "limit_days": RECENT_SESSIONS,
            "handled": "failed",
        }
        _append(day, REBUILD_CHECK, Severity.WARNING, failed)
        return f"rebuild failed, see {REBUILD_CHECK}"


def _rebuild_covered(day: date) -> str:
    return _rebuild(day, build=False)


STEPS: dict[str, Callable[[date], str]] = {
    "rebuild": _rebuild,
    "rebuild_covered": _rebuild_covered,
    "labels": _labels,
    "features": _features,
    "recent_features": _recent_features,
    "coverage": _coverage,
    "predict": _predict,
    "monitor": _monitor,
    "promote": _promote,
}


def _record(day: date, check: str, severity: str, detail: dict[str, Any]) -> None:
    """One row per day and check, replaced on rerun; warnings are sent as alerts."""
    with transaction.atomic():
        DataQualityLog.objects.filter(date=day, check_name=check).delete()
        row = DataQualityLog.objects.create(
            date=day, check_name=check, severity=severity, detail=detail
        )
    if severity != Severity.INFO:
        dispatch([row])


def _finished(day: date, check: str) -> bool:
    return DataQualityLog.objects.filter(
        date=day, check_name=check, severity=Severity.INFO
    ).exists()


def _us_close_landed(day: date) -> bool:
    return DataQualityLog.objects.filter(date=day, check_name=US_CLOSE_CHECK).exists()


def _early(slot: str) -> bool:
    """A run fired before its slot, as beat does on waking past midnight, is skipped and logged."""
    now = timezone.localtime()
    if now.time() >= time.fromisoformat(settings.RESEARCH_AT[slot]):
        return False
    logger.warning("research skipped %s: fired at %s, before its slot", slot, now.isoformat())
    return True


def _concurrency(task: Task[Any, Any]) -> Any:
    """The worker's configured process count; the schedule runs only when it is exactly 1."""
    return task.app.conf.get("worker_concurrency")


def _with_rebuild(names: tuple[str, ...]) -> tuple[str, ...]:
    """A last step for waiting adj_factor changes: recorded if the chain's steps cover them."""
    if not _open_notes():
        return names
    covered = "recent_features" in names and "labels" in names
    return (*names, "rebuild_covered" if covered else "rebuild")


def _failed(check: str, day: date, name: str, error: Exception, done: list[Done]) -> None:
    logger.error("research %s %s failed: %s", name, day, error)
    detail = {"failed": name, "error": f"{type(error).__name__}: {error}", "steps": done}
    _record(day, check, Severity.WARNING, detail)


def _start(check: str, day: date, names: tuple[str, ...], extra: dict[str, str]) -> None:
    """Queue the steps as one chain; each starts only once the one before it succeeded."""
    iso = day.isoformat()
    first, *rest = names
    chain(
        step.s([], first, check, iso),
        *(step.s(name, check, iso) for name in rest),
        finish.s(check, iso, extra),
    ).delay()


@shared_task(bind=True, max_retries=None, name="research.step")
def step(self: Task[Any, Any], done: list[Done], name: str, check: str, day: str) -> list[Done]:
    """One chain step under the research lock; a busy lock retries, a failure stops the chain."""
    session = date.fromisoformat(day)
    try:
        with research_lock(f"{name} {day}"):
            summary = STEPS[name](session)
    except ResearchBusyError as error:
        if self.request.retries < settings.RESEARCH_LOCK_RETRIES:
            raise self.retry(exc=error, countdown=settings.RESEARCH_LOCK_COUNTDOWN) from error
        _failed(check, session, name, error, done)
        raise
    except Exception as error:
        _failed(check, session, name, error, done)
        raise
    return [*done, {"step": name, "result": summary}]


@shared_task(name="research.finish")
def finish(done: list[Done], check: str, day: str, extra: dict[str, str]) -> str:
    """Record the chain as done, as a warning without the US close; a morning may then retrain."""
    session = date.fromisoformat(day)
    severity = Severity.WARNING if extra else Severity.INFO
    _record(session, check, severity, {"steps": done, **extra})
    if check == MORNING_CHECK:
        _retrain_after(session)
    return str(severity)


def _retrain_detail(
    refused: dict[str, str],
    built: dict[str, str] | None = None,
    failed: dict[str, str] | None = None,
    free: float | None = None,
    runs: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    return {
        "built": built or {},
        "refused": refused,
        "failed": failed or {},
        "free_percent": free,
        "runs": runs or {},
    }


def _retrain_hours() -> bool:
    """From the morning slot to the retrain cutoff, the hours a retrain may start in."""
    now = timezone.localtime().time()
    start = time.fromisoformat(settings.RESEARCH_AT["MORNING"])
    return start <= now < time.fromisoformat(settings.RESEARCH_RETRAIN_BEFORE)


def _retrain_after(day: date) -> None:
    """Queue the month's one retrain from a morning done in time among its first sessions."""
    if _month(RETRAIN_CHECK, day).exists():
        return
    opening = TradingDays.from_db().between(day.replace(day=1), day)
    if len(opening) > RETRAIN_WINDOW:
        skipped = _retrain_detail(dict.fromkeys(HoldoutScope.values, SKIPPED))
        _record(day, RETRAIN_CHECK, Severity.WARNING, skipped)
    elif _retrain_hours():
        retrain.delay(day.isoformat())


@shared_task(bind=True, max_retries=None, name="research.retrain")
def retrain(self: Task[Any, Any], day: str) -> str:
    """This month's N and N′ of every frozen scope as of day, once memory and labels allow."""
    if not settings.RESEARCH_SCHEDULE_ENABLED:
        return "disabled"
    session = date.fromisoformat(day)
    if _month(RETRAIN_CHECK, session).exists():
        return "done"
    if not _retrain_hours():
        logger.warning("research retrain %s not started: outside its hours", day)
        return "late"
    concurrency = _concurrency(self)
    if concurrency != 1:
        refused = dict.fromkeys(HoldoutScope.values, f"worker concurrency {concurrency}, not 1")
        _record(session, RETRAIN_CHECK, Severity.WARNING, _retrain_detail(refused))
        return "concurrent"
    free = free_percent()
    if free is not None and free < settings.RESEARCH_RETRAIN_MIN_FREE:
        if self.request.retries < settings.RESEARCH_RETRAIN_RETRIES:
            logger.warning("research retrain %s deferred: %.0f%% of memory free", day, free)
            raise self.retry(args=(day,), countdown=settings.RESEARCH_RETRAIN_COUNTDOWN)
        short = f"{free:.0f}% of memory free, under {settings.RESEARCH_RETRAIN_MIN_FREE}%"
        detail = _retrain_detail(dict.fromkeys(HoldoutScope.values, short), free=free)
        _record(session, RETRAIN_CHECK, Severity.WARNING, detail)
        return "short of memory"
    detail = _build_all(session, free)
    severity = Severity.WARNING if detail["refused"] or detail["failed"] else Severity.INFO
    _record(session, RETRAIN_CHECK, severity, detail)
    return str(severity)


def _built_this_month(scope: str, day: date) -> bool:
    """A production build of scope made this month under a key other than this retrain's."""
    own = retrain_key(scope, day)
    made = ModelRun.objects.filter(
        run_key__startswith=f"{scope}-",
        params__has_key=PRODUCTION,
        created_at__date__gte=day.replace(day=1),
    )
    return made.exclude(run_key__in=[own, shadow_key(own)]).exists()


def _build_all(day: date, free: float | None) -> dict[str, Any]:
    """Build each frozen scope in turn, timing each and noting the worker's peak memory after."""
    last = LabelDaily.objects.aggregate(last=Max("date"))["last"]
    built: dict[str, str] = {}
    refused: dict[str, str] = {}
    failed: dict[str, str] = {}
    runs: dict[str, dict[str, float]] = {}
    for scope in HoldoutScope.values:
        research = DESIGNS[scope].research_run_key
        if last is None or last < day:
            refused[scope] = f"label_daily ends on {last}, before {day}"
        elif research is None:
            refused[scope] = NOT_FROZEN
        elif _built_this_month(scope, day):
            refused[scope] = ALREADY_BUILT
        else:
            key = retrain_key(scope, day)
            start = monotonic()
            try:
                build_run(scope, key, research, day, settings.FEATURE_PARQUET_DIR)
            except Exception as error:
                logger.error("research retrain %s %s failed: %s", scope, day, error)
                failed[scope] = f"{type(error).__name__}: {error}"
            else:
                built[scope] = key
            elapsed = round(monotonic() - start, 1)
            runs[scope] = {"elapsed_s": elapsed, "peak_rss_mb": round(peak_rss_mb(), 1)}
    return _retrain_detail(refused, built, failed, free, runs)


def _flag_missed(calendar: TradingDays, day: date) -> None:
    """Warn when the session before day had no morning run though one of the few before it did."""
    try:
        before = calendar.shift(day, -1)
        since = calendar.shift(before, -MISSED_LOOKBACK)
    except OutsideCalendarError:
        return
    seen = DataQualityLog.objects.filter(check_name=MORNING_CHECK)
    if seen.filter(date=before).exists():
        return
    if not seen.filter(date__gte=since, date__lt=before).exists():
        return
    _record(before, MORNING_CHECK, Severity.WARNING, {"reason": MISSED})


@shared_task(bind=True, max_retries=None, name="research.evening")
def evening(self: Task[Any, Any], day: str | None = None) -> str:
    """The last sixty sessions' features and their labels, once the research gate passes."""
    if not settings.RESEARCH_SCHEDULE_ENABLED:
        return "disabled"
    if day is None and _early("EVENING"):
        return "early"
    session = timezone.localtime().date() if day is None else date.fromisoformat(day)
    if not TradingDays.from_db().is_trading_day(session):
        return "closed"
    if _finished(session, EVENING_CHECK):
        return "done"
    concurrency = _concurrency(self)
    if concurrency != 1:
        detail = {"reason": CONCURRENT, "concurrency": concurrency}
        _record(session, EVENING_CHECK, Severity.WARNING, detail)
        return "concurrent"
    if not ready_for_research(session):
        if self.request.retries < settings.ETL_NOT_READY_RETRIES:
            raise self.retry(
                args=(session.isoformat(),), countdown=settings.ETL_NOT_READY_COUNTDOWN
            )
        _record(session, EVENING_CHECK, Severity.WARNING, {"reason": GATE_SHUT})
        return "not ready"
    _start(EVENING_CHECK, session, _with_rebuild(EVENING_STEPS), {})
    return "started"


@shared_task(bind=True, max_retries=None, name="research.morning")
def morning(self: Task[Any, Any], day: str | None = None) -> str:
    """Before the open: features with the US close, predictions, monitor, then pending swaps."""
    if not settings.RESEARCH_SCHEDULE_ENABLED:
        return "disabled"
    if day is None and _early("MORNING"):
        return "early"
    if day is None:
        calendar = TradingDays.from_db()
        session = calendar.latest_on_or_before(timezone.localtime().date() - timedelta(days=1))
        _flag_missed(calendar, session)
    else:
        session = date.fromisoformat(day)
    if _finished(session, MORNING_CHECK):
        return "done"
    concurrency = _concurrency(self)
    if concurrency != 1:
        detail = {"reason": CONCURRENT, "concurrency": concurrency}
        _record(session, MORNING_CHECK, Severity.WARNING, detail)
        return "concurrent"
    if not ready_for_research(session):
        _record(session, MORNING_CHECK, Severity.WARNING, {"reason": GATE_SHUT})
        return "gate shut"
    landed = _us_close_landed(session)
    if not landed and self.request.retries < settings.RESEARCH_US_CLOSE_RETRIES:
        raise self.retry(
            args=(session.isoformat(),), countdown=settings.RESEARCH_US_CLOSE_COUNTDOWN
        )
    made_up = ("labels", *MORNING_STEPS, "recent_features")
    steps = MORNING_STEPS if _finished(session, EVENING_CHECK) else made_up
    last = ("promote",) if _pending(session) else ()
    names = _with_rebuild((*steps, *last))
    _start(MORNING_CHECK, session, names, {} if landed else {"us_close": US_CLOSE_MISSING})
    return "started"
