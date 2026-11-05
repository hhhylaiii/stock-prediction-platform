"""The header strip: latest session, data and prediction dates, quality and degraded models."""

from django.db.models import Max
from django.utils import timezone
from rest_framework.request import Request
from rest_framework.response import Response

from apps.api.envelope import envelope
from apps.api.ops import CHECK_FIELDS, session_summary
from apps.api.serving import DEFAULT_HORIZON, Slot, latest_served_date, runs_on
from apps.api.sessions import latest_session, load_calendar
from apps.api.views import ReadOnlyView
from apps.etl.models import DataQualityLog, EtlRun
from apps.market.models import DailyPrice
from apps.research.models import OUTPUT_HORIZONS, ModelObjective, ModelStatus

ALL_SLOTS = [
    Slot(objective, horizon) for objective in ModelObjective.values for horizon in OUTPUT_HORIZONS
]


class StatusView(ReadOnlyView):
    def get(self, request: Request) -> Response:
        session = latest_session(load_calendar())
        price_date = DailyPrice.objects.aggregate(latest=Max("date"))["latest"]
        prediction_date = latest_served_date(DEFAULT_HORIZON)
        data_date = price_date or session
        logs = list(
            DataQualityLog.objects.filter(date=data_date)
            .order_by("check_name", "id")
            .values("date", *CHECK_FIELDS)
        )
        summary = session_summary(data_date, logs)
        updated_at = EtlRun.objects.filter(
            start_date=data_date, end_date=data_date, status=EtlRun.Status.SUCCESS
        ).aggregate(latest=Max("finished_at"))["latest"]
        ready = (
            price_date is not None and prediction_date is not None and prediction_date >= price_date
        )
        serving = {} if prediction_date is None else runs_on(prediction_date, ALL_SLOTS)
        degraded = sorted(
            (run for run in serving.values() if run.status == ModelStatus.DEGRADED),
            key=lambda run: (run.objective, run.horizon),
        )
        return envelope(
            {
                "session": session,
                "price_date": price_date,
                "prediction_date": prediction_date,
                "predictions_ready": ready,
                "quality": {
                    key: summary[key] for key in ("date", "passed", "worst", "missing_checks")
                },
                "updated_at": None if updated_at is None else timezone.localtime(updated_at),
                "degraded": [
                    {"run_id": run.run_id, "objective": run.objective, "horizon": run.horizon}
                    for run in degraded
                ],
            }
        )
