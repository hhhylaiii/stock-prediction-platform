"""Matured predictions of one stock against what happened, one every horizon sessions."""

from collections import defaultdict
from datetime import date
from typing import Any

from django.db.models import F, Q
from rest_framework.request import Request
from rest_framework.response import Response

from apps.api.envelope import envelope
from apps.api.pagination import EnvelopePagination
from apps.api.params import HistoryQuery, parse
from apps.api.pricing import choose_interval
from apps.api.serving import (
    SERVING_FIELDS,
    ServingRun,
    Slot,
    best_runs,
    eligible,
    roles_for,
    serving_run,
)
from apps.api.sessions import load_calendar
from apps.api.stocks import require_stock
from apps.api.views import ReadOnlyView
from apps.market.calendar import CalendarError
from apps.research.models import FeatureDaily, LabelDaily, ModelStatus

HISTORY_ROLES = ("return", "interval", "volatility")
ROW_FIELDS = ("date", "ret_pred", "q10", "q50", "q90", "vol_pred", "in_universe")
RATIO_DIGITS = 6


def matured(stock_id: str, horizon: int) -> dict[date, float]:
    """Returns from the t+1 open to the horizon close, the label the return models learn."""
    rows = LabelDaily.objects.filter(
        stock_id=stock_id, horizon=horizon, fwd_ret_open__isnull=False
    ).values_list("date", "fwd_ret_open")
    return {day: value for day, value in rows if value is not None}


def predicted(
    stock_id: str, slots: list[Slot], first: date, last: date
) -> dict[date, list[dict[str, Any]]]:
    """Out-of-sample rows only: a day inside a run's training window is not a past forecast."""
    condition = Q()
    for slot in slots:
        condition |= Q(run__objective=slot.objective, horizon=slot.horizon)
    rows = (
        eligible()
        .filter(condition, stock_id=stock_id, date__range=(first, last))
        .filter(date__gt=F("run__train_end"))
        .values(*SERVING_FIELDS, *ROW_FIELDS)
    )
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["date"]].append(row)
    return grouped


def tech_vols(stock_id: str, first: date, last: date) -> dict[date, float]:
    rows = FeatureDaily.objects.filter(
        stock_id=stock_id, date__range=(first, last), tech_vol_20d__isnull=False
    ).values_list("date", "tech_vol_20d")
    return {day: value for day, value in rows if value is not None}


def outcome(
    day: date,
    actual: float,
    rows: list[dict[str, Any]],
    roles: dict[str, Slot],
    horizon: int,
    tech_vol: float | None,
) -> dict[str, Any] | None:
    """The served return and band of the day judged against the realised return."""
    runs = best_runs((serving_run(row) for row in rows), roles.values())
    by_run = {row["run_id"]: row for row in rows}

    def part(role: str) -> tuple[ServingRun | None, dict[str, Any]]:
        run = runs.get(roles[role]) if role in roles else None
        return run, ({} if run is None else by_run[run.run_id])

    ret_run, ret_row = part("return")
    band_run, band_row = part("interval")
    ret = ret_row.get("ret_pred") if ret_row.get("ret_pred") is not None else ret_row.get("q50")
    if ret_run is None or ret is None:
        return None
    interval = choose_interval(
        1.0,
        horizon,
        ret=ret,
        q10=band_row.get("q10"),
        q90=band_row.get("q90"),
        vol=part("volatility")[1].get("vol_pred"),
        tech_vol=tech_vol,
        quantile_alert=band_run is not None and band_run.status == ModelStatus.DEGRADED,
    )
    band = None
    if interval is not None:
        band = {
            "ret_low": round(interval.ret_low, RATIO_DIGITS),
            "ret_high": round(interval.ret_high, RATIO_DIGITS),
        }
    return {
        "date": day,
        "ret_pred": ret,
        "actual": actual,
        "interval": band,
        "interval_source": None if interval is None else interval.source,
        "inside": None if interval is None else interval.ret_low <= actual <= interval.ret_high,
        "in_universe": ret_row["in_universe"],
        "run_id": ret_run.run_id,
    }


def spaced(outcomes: list[dict[str, Any]], horizon: int) -> list[dict[str, Any]]:
    """Newest first, each pick at least horizon sessions before the previous one."""
    if not outcomes:
        return []
    calendar = load_calendar()
    chosen: list[dict[str, Any]] = []
    bound: date | None = None
    for item in outcomes:
        if bound is not None and item["date"] > bound:
            continue
        chosen.append(item)
        try:
            bound = calendar.shift(calendar.latest_on_or_before(item["date"]), -horizon)
        except CalendarError:
            break
    return chosen


class StockHistoryView(ReadOnlyView):
    def get(self, request: Request, stock_id: str) -> Response:
        horizon = parse(HistoryQuery, request)["horizon"]
        require_stock(stock_id)
        roles = {role: slot for role, slot in roles_for(horizon).items() if role in HISTORY_ROLES}
        actuals = matured(stock_id, horizon)
        rows, vols = {}, {}
        if actuals:
            first, last = min(actuals), max(actuals)
            rows = predicted(stock_id, list(roles.values()), first, last)
            vols = tech_vols(stock_id, first, last)
        outcomes = [
            item
            for day in sorted(actuals, reverse=True)
            if (
                item := outcome(day, actuals[day], rows.get(day, []), roles, horizon, vols.get(day))
            )
        ]
        history = spaced(outcomes, horizon)
        paginator = EnvelopePagination()
        page = paginator.paginate_queryset(history, request, view=self) or []
        meta = {
            "horizon": horizon,
            "inside": sum(item["inside"] is True for item in history),
            "with_interval": sum(item["inside"] is not None for item in history),
            **paginator.meta(),
        }
        return envelope(page, meta)
