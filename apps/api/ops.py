"""Operations endpoints: ETL runs of a day and the daily data-quality checks."""

from datetime import date
from itertools import groupby
from operator import itemgetter
from typing import Any, NoReturn

from django.http import HttpRequest
from rest_framework import serializers
from rest_framework.exceptions import NotFound
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.api.envelope import envelope
from apps.api.pagination import EnvelopePagination
from apps.api.params import QualityQuery, RunQuery, parse
from apps.api.sessions import latest_session, load_calendar, trailing_sessions
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
    """A session's checks; ready means every check ran and none is an error, like the gate."""
    ran = {entry["check_name"] for entry in logs}
    missing = [check for check in CHECKS if check not in ran]
    worst = max((entry["severity"] for entry in logs), key=SEVERITY_ORDER.index, default=None)
    return {
        "date": day,
        "ready": not missing and worst != Severity.ERROR,
        "worst": worst,
        "missing_checks": missing,
        "checks": [{field: entry[field] for field in CHECK_FIELDS} for entry in logs],
    }


class EtlRunListView(APIView):
    def get(self, request: Request) -> Response:
        day = parse(RunQuery, request).get("date") or latest_session(load_calendar())
        runs = EtlRun.objects.filter(start_date__lte=day, end_date__gte=day).order_by(
            "started_at", "run_id"
        )
        paginator = EnvelopePagination()
        page = paginator.paginate_queryset(runs, request, view=self) or []
        data = EtlRunSerializer(page, many=True).data
        return envelope(data, {"date": day, **paginator.meta()})


class DataQualityView(APIView):
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


class NotFoundView(APIView):
    """Any other path under /api answers a not-found envelope for every method."""

    def http_method_not_allowed(self, request: HttpRequest, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotFound()
