"""Why the rank model placed a stock where it did: top contributions, the rest, raw values."""

from datetime import date
from typing import Any

from rest_framework.request import Request
from rest_framework.response import Response

from apps.api.envelope import envelope
from apps.api.feature_text import LABELS
from apps.api.params import FeatureQuery, parse
from apps.api.serving import ServingRun, Slot, latest_served_date, runs_on
from apps.api.stocks import require_stock
from apps.api.views import ReadOnlyView
from apps.research.features.registry import FEATURES
from apps.research.models import REST_FEATURE, FeatureDaily, ModelObjective, PredictionContrib

GROUPS = {spec.name: spec.group.value for spec in FEATURES}
STORED = frozenset(field.name for field in FeatureDaily._meta.get_fields()) & set(GROUPS)
CONTRIB_FIELDS = ("position", "feature", "contrib", "n_features")


def serving_rank_run(day: date | None, horizon: int) -> ServingRun | None:
    if day is None:
        return None
    slot = Slot(ModelObjective.RANK, horizon)
    return runs_on(day, [slot]).get(slot)


def feature_values(stock_id: str, day: date, names: list[str]) -> dict[str, Any]:
    """Raw first-batch values from feature_daily; later batches live outside the database."""
    columns = [name for name in names if name in STORED]
    if not columns:
        return {}
    row = FeatureDaily.objects.filter(stock_id=stock_id, date=day).values(*columns).first()
    return row or {}


def feature_item(row: dict[str, Any], value: Any) -> dict[str, Any]:
    """An unknown feature keeps its code as the label and has no unit or group."""
    feature = row["feature"]
    label, unit = LABELS.get(feature, (feature, None))
    return {
        "position": row["position"],
        "feature": feature,
        "label": label,
        "unit": unit,
        "group": GROUPS.get(feature),
        "contrib": row["contrib"],
        "value": value,
    }


def contributions(stock_id: str, day: date, run: ServingRun) -> dict[str, Any]:
    rows = list(
        PredictionContrib.objects.filter(
            run_id=run.run_id, stock_id=stock_id, date=day, horizon=run.horizon
        )
        .order_by("position")
        .values(*CONTRIB_FIELDS)
    )
    top = [row for row in rows if row["feature"] != REST_FEATURE]
    rest = next((row for row in rows if row["feature"] == REST_FEATURE), None)
    values = feature_values(stock_id, day, [row["feature"] for row in top])
    return {
        "items": [feature_item(row, values.get(row["feature"])) for row in top],
        "rest": None
        if rest is None
        else {"contrib": rest["contrib"], "n_features": rest["n_features"]},
        "total": sum(row["contrib"] for row in rows) if rows else None,
        "n_features": len(top) + (0 if rest is None else rest["n_features"]),
    }


EMPTY: dict[str, Any] = {"items": [], "rest": None, "total": None, "n_features": 0}


class StockFeaturesView(ReadOnlyView):
    def get(self, request: Request, stock_id: str) -> Response:
        query = parse(FeatureQuery, request)
        require_stock(stock_id)
        horizon = query["horizon"]
        day = query.get("date") or latest_served_date(horizon)
        run = serving_rank_run(day, horizon)
        data = EMPTY if run is None or day is None else contributions(stock_id, day, run)
        meta = {
            "date": day,
            "horizon": horizon,
            "run_id": None if run is None else run.run_id,
            "model_type": None if run is None else run.model_type,
            "version": None if run is None else run.version,
        }
        return envelope(data, meta)
