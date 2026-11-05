"""One stock's merged predictions for both horizons, with its realized volatility."""

from datetime import date
from typing import Any

from rest_framework.request import Request
from rest_framework.response import Response

from apps.api.envelope import envelope
from apps.api.params import DateQuery, parse
from apps.api.predictions import (
    BASE_FIELDS,
    degraded_roles,
    merged_rows,
    rank_rows,
    served_models,
    with_closes,
)
from apps.api.serving import latest_served_date, roles_for, runs_on
from apps.api.stocks import require_stock
from apps.api.views import ReadOnlyView
from apps.research.models import OUTPUT_HORIZONS, FeatureDaily

REALIZED_FIELDS = ("date", "tech_vol_20d", "tech_vol_60d")
REALIZED_HORIZON = 20


def horizon_entry(stock_id: str, horizon: int, asked: date | None) -> dict[str, Any]:
    day = asked or latest_served_date(horizon)
    roles = roles_for(horizon)
    runs = {} if day is None else runs_on(day, roles.values())
    base = rank_rows(day, horizon, roles, runs)
    own = with_closes(base.filter(stock_id=stock_id), day or date.min).values(*BASE_FIELDS)
    rows = merged_rows(day, horizon, roles, runs, list(own))
    return {
        "horizon": horizon,
        "date": day,
        "universe": base.filter(in_universe=True).count(),
        "prediction": rows[0] if rows else None,
        "models": served_models(roles, runs),
        "degraded": degraded_roles(roles, runs),
    }


def realized_vol(stock_id: str, day: date | None) -> dict[str, Any] | None:
    """Trailing volatility features on the last feature day up to the 20-day prediction."""
    if day is None:
        return None
    row: dict[str, Any] | None = (
        FeatureDaily.objects.filter(stock_id=stock_id, date__lte=day)
        .order_by("-date")
        .values(*REALIZED_FIELDS)
        .first()
    )
    return row


class StockPredictionsView(ReadOnlyView):
    def get(self, request: Request, stock_id: str) -> Response:
        query = parse(DateQuery, request)
        require_stock(stock_id)
        entries = [horizon_entry(stock_id, h, query.get("date")) for h in OUTPUT_HORIZONS]
        by_horizon = {entry["horizon"]: entry["date"] for entry in entries}
        anchor = by_horizon[REALIZED_HORIZON] or by_horizon[min(OUTPUT_HORIZONS)]
        return envelope(
            {
                "stock_id": stock_id,
                "horizons": entries,
                "realized_vol": realized_vol(stock_id, anchor),
            }
        )
