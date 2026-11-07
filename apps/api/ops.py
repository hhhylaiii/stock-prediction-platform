"""Operations endpoints: ETL runs of a day, the daily data-quality checks and corrections."""

from datetime import date
from itertools import groupby
from operator import itemgetter
from typing import Any, NoReturn

from django.conf import settings
from django.core.cache import cache
from django.db.models import Case, Count, IntegerField, Max, Min, QuerySet, Value, When
from django.db.models.fields.json import KeyTextTransform
from django.http import HttpRequest
from rest_framework import serializers
from rest_framework.exceptions import NotFound
from rest_framework.request import Request
from rest_framework.response import Response

from apps.api.envelope import envelope
from apps.api.pagination import EnvelopePagination
from apps.api.params import QualityQuery, RunQuery, parse
from apps.api.sessions import latest_session, load_calendar, trailing_sessions
from apps.api.views import ReadOnlyView
from apps.etl.corrections import CORRECTION_CHECKS
from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.quality import CHECKS
from apps.research.evaluation import ALERT_SHARE

Severity = DataQualityLog.Severity
SEVERITY_ORDER = (Severity.INFO, Severity.WARNING, Severity.ERROR)
CHECK_FIELDS = ("check_name", "severity", "detail")
MODEL_CHECK_FIELDS = ("id", *CHECK_FIELDS)
MODEL_CHECK_PREFIXES = ("output_", "monitor_", "model_", "research_")
CORRECTIONS_CACHE_KEY = "api:data-quality:corrections"


class EtlRunSerializer(serializers.ModelSerializer[EtlRun]):
    class Meta:
        model = EtlRun
        fields = (
            "run_id",
            "source",
            "dataset",
            "start_date",
            "end_date",
            "status",
            "rows",
            "error",
            "started_at",
            "finished_at",
        )


def session_summary(day: date, logs: list[dict[str, Any]]) -> dict[str, Any]:
    """Data checks set the worst; any error still fails the day, as the prediction gate rules."""
    model = [entry for entry in logs if entry["check_name"].startswith(MODEL_CHECK_PREFIXES)]
    data = [
        entry
        for entry in logs
        if not entry["check_name"].startswith(MODEL_CHECK_PREFIXES)
        and entry["check_name"] not in CORRECTION_CHECKS
    ]
    ran = {entry["check_name"] for entry in data}
    missing = [check for check in CHECKS if check not in ran]
    worst = max((entry["severity"] for entry in data), key=SEVERITY_ORDER.index, default=None)
    return {
        "date": day,
        "passed": not missing and all(entry["severity"] != Severity.ERROR for entry in logs),
        "worst": worst,
        "missing_checks": missing,
        "checks": [{field: entry[field] for field in CHECK_FIELDS} for entry in data],
        "model_checks": [{field: entry[field] for field in MODEL_CHECK_FIELDS} for entry in model],
    }


def warning_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    """Checks with a warning, one per name however many rows it wrote, and those rows."""
    warned = [check["check_name"] for check in checks if check["severity"] == Severity.WARNING]
    return {"warnings": len(set(warned)), "warning_rows": len(warned)}


def runs_of(day: date, scope: str) -> QuerySet[EtlRun]:
    """Single-day runs of the day, or with scope covering every run whose range holds it."""
    if scope == "day":
        runs = EtlRun.objects.filter(start_date=day, end_date=day)
    else:
        runs = EtlRun.objects.filter(start_date__lte=day, end_date__gte=day)
    return runs.order_by("started_at", "run_id")


class EtlRunListView(ReadOnlyView):
    def get(self, request: Request) -> Response:
        query = parse(RunQuery, request)
        day = query.get("date") or latest_session(load_calendar())
        paginator = EnvelopePagination()
        page = paginator.paginate_queryset(runs_of(day, query["scope"]), request, view=self) or []
        data = EtlRunSerializer(page, many=True).data
        return envelope(data, {"date": day, "scope": query["scope"], **paginator.meta()})


class DataQualityView(ReadOnlyView):
    def get(self, request: Request) -> Response:
        count = parse(QualityQuery, request)["days"]
        calendar = load_calendar()
        sessions = trailing_sessions(calendar, latest_session(calendar), count)
        logs = (
            DataQualityLog.objects.filter(date__range=(sessions[0], sessions[-1]))
            .order_by("date", "check_name", "id")
            .values("date", *MODEL_CHECK_FIELDS)
        )
        by_day = {day: list(entries) for day, entries in groupby(logs, key=itemgetter("date"))}
        data = [session_summary(day, by_day.get(day, [])) for day in reversed(sessions)]
        meta = {
            "days": count,
            "start_date": sessions[0],
            "end_date": sessions[-1],
            "checks": list(CHECKS),
            "output_alert_share": ALERT_SHARE,
        }
        return envelope(data, meta)


def corrections() -> list[dict[str, Any]]:
    """Each correction check over all its rows: worst severity, rows, stocks and date span."""
    rank = Case(
        *(When(severity=severity, then=Value(n)) for n, severity in enumerate(SEVERITY_ORDER)),
        output_field=IntegerField(),
    )
    rows = (
        DataQualityLog.objects.filter(check_name__in=CORRECTION_CHECKS)
        .values("check_name")
        .annotate(
            worst=Max(rank),
            rows=Count("id"),
            stocks=Count(KeyTextTransform("stock_id", "detail"), distinct=True),
            first_date=Min("date"),
            last_date=Max("date"),
        )
        .order_by("check_name")
    )
    return [
        {
            "check_name": row["check_name"],
            "severity": SEVERITY_ORDER[row["worst"]],
            "rows": row["rows"],
            "stocks": row["stocks"],
            "first_date": row["first_date"],
            "last_date": row["last_date"],
        }
        for row in rows
    ]


class CorrectionsView(ReadOnlyView):
    def get(self, request: Request) -> Response:
        seconds = settings.API_CORRECTIONS_CACHE_SECONDS
        data = cache.get_or_set(CORRECTIONS_CACHE_KEY, corrections, seconds)
        meta = {"checks": sorted(CORRECTION_CHECKS), "cache_seconds": seconds}
        return envelope(data, meta)


class NotFoundView(ReadOnlyView):
    """Any other path under /api answers a not-found envelope for every method."""

    def http_method_not_allowed(self, request: HttpRequest, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotFound()
