"""The header strip: latest session, data and prediction dates, quality and degraded models."""

from datetime import date
from typing import Any

from django.db.models import Max
from django.utils import timezone
from rest_framework.request import Request
from rest_framework.response import Response

from apps.api.envelope import envelope
from apps.api.ops import MODEL_CHECK_FIELDS, session_summary, warning_counts
from apps.api.serving import Slot, latest_days, runs_on_days
from apps.api.sessions import latest_session, load_calendar
from apps.api.views import ReadOnlyView
from apps.etl.models import DataQualityLog, EtlRun
from apps.market.models import DailyPrice
from apps.research.models import OUTPUT_HORIZONS, ModelObjective, ModelStatus

ALL_SLOTS = [
    Slot(objective, horizon) for objective in ModelObjective.values for horizon in OUTPUT_HORIZONS
]


def horizon_entries(days: dict[Slot, date], price_date: date | None) -> list[dict[str, Any]]:
    """Each horizon's latest day of a serving rank run, and whether it has caught up with prices."""
    entries = []
    for horizon in sorted(OUTPUT_HORIZONS):
        day = days.get(Slot(ModelObjective.RANK, horizon))
        ready = price_date is not None and day is not None and day >= price_date
        entries.append({"horizon": horizon, "prediction_date": day, "ready": ready})
    return entries


class StatusView(ReadOnlyView):
    def get(self, request: Request) -> Response:
        session = latest_session(load_calendar())
        price_date = DailyPrice.objects.aggregate(latest=Max("date"))["latest"]
        days = latest_days(ALL_SLOTS)
        horizons = horizon_entries(days, price_date)
        predicted = [entry["prediction_date"] for entry in horizons if entry["prediction_date"]]
        data_date = price_date or session
        logs = list(
            DataQualityLog.objects.filter(date=data_date)
            .order_by("check_name", "id")
            .values("date", *MODEL_CHECK_FIELDS)
        )
        summary = session_summary(data_date, logs)
        updated_at = EtlRun.objects.filter(
            start_date=data_date, end_date=data_date, status=EtlRun.Status.SUCCESS
        ).aggregate(latest=Max("finished_at"))["latest"]
        degraded = sorted(
            (run for run in runs_on_days(days).values() if run.status == ModelStatus.DEGRADED),
            key=lambda run: (run.objective, run.horizon),
        )
        return envelope(
            {
                "session": session,
                "price_date": price_date,
                "prediction_date": max(predicted, default=None),
                "predictions_ready": any(entry["ready"] for entry in horizons),
                "horizons": horizons,
                "quality": {
                    **{key: summary[key] for key in ("date", "passed", "worst", "missing_checks")},
                    **warning_counts(summary["checks"]),
                },
                "updated_at": None if updated_at is None else timezone.localtime(updated_at),
                "degraded": [
                    {"run_id": run.run_id, "objective": run.objective, "horizon": run.horizon}
                    for run in degraded
                ],
            }
        )
