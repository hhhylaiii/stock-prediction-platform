"""Operations endpoints: ETL runs of a day and the daily data-quality checks."""

from datetime import date
from itertools import groupby
from operator import itemgetter
from typing import Any, NoReturn

from django.db.models import QuerySet
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
from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.quality import CHECKS

Severity = DataQualityLog.Severity
SEVERITY_ORDER = (Severity.INFO, Severity.WARNING, Severity.ERROR)
CHECK_FIELDS = ("check_name", "severity", "detail")


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
    """A session's checks; passed means every check ran and none is an error, as the gate rules."""
    ran = {entry["check_name"] for entry in logs}
    missing = [check for check in CHECKS if check not in ran]
    worst = max((entry["severity"] for entry in logs), key=SEVERITY_ORDER.index, default=None)
    return {
        "date": day,
        "passed": not missing and worst != Severity.ERROR,
        "worst": worst,
        "missing_checks": missing,
        "checks": [{field: entry[field] for field in CHECK_FIELDS} for entry in logs],
    }


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
            .values("date", *CHECK_FIELDS)
        )
        by_day = {day: list(entries) for day, entries in groupby(logs, key=itemgetter("date"))}
        data = [session_summary(day, by_day.get(day, [])) for day in reversed(sessions)]
        meta = {
            "days": count,
            "start_date": sessions[0],
            "end_date": sessions[-1],
            "checks": list(CHECKS),
        }
        return envelope(data, meta)


class NotFoundView(ReadOnlyView):
    """Any other path under /api answers a not-found envelope for every method."""

    def http_method_not_allowed(self, request: HttpRequest, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotFound()
