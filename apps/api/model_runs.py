"""The serving models with recent monitoring, research totals and the monthly rank IC chart."""

import calendar as months
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from django.conf import settings
from django.db.models import Avg, Count, Q, StdDev
from rest_framework.request import Request
from rest_framework.response import Response

from apps.api.envelope import envelope
from apps.api.params import MonthlyIcQuery, parse
from apps.api.serving import (
    SERVING,
    ServingRun,
    Slot,
    latest_served_date,
    roles_for,
    runs_on,
)
from apps.api.views import ReadOnlyView
from apps.research.eval_report import RANK_METRICS
from apps.research.models import (
    OUTPUT_HORIZONS,
    BacktestMetric,
    MetricGrain,
    ModelMonitor,
    ModelObjective,
)

PAGE_ORDER = (
    Slot(ModelObjective.RANK, 20),
    Slot(ModelObjective.RANK, 5),
    Slot(ModelObjective.REGRESSION, 5),
    Slot(ModelObjective.QUANTILE, 20),
    Slot(ModelObjective.VOLATILITY, 20),
    Slot(ModelObjective.CLASSIFICATION, 20),
)
RECENT_MONTHS = 6
DIGITS = 4
MONITOR_MEANS = ("ic", "coverage", "width", "brier", "auc", "direction_acc", "mae")
BASELINE_MODEL = "ridge"
MONTH_IC, MONTH_DAYS = RANK_METRICS[0], RANK_METRICS[-1]
ENTRY_FIELDS = ("run_id", "run_key", "model_type", "version", "status", "train_start", "train_end")


def months_before(day: date, count: int) -> date:
    """The same day count months earlier, clamped to that month's last day."""
    year, month = divmod(day.year * 12 + day.month - 1 - count, 12)
    return date(year, month + 1, min(day.day, months.monthrange(year, month + 1)[1]))


def serving_models() -> list[tuple[ServingRun, date]]:
    """Per page slot the active or degraded run of each horizon's latest served day."""
    found: dict[Slot, tuple[ServingRun, date]] = {}
    for horizon in sorted(OUTPUT_HORIZONS, reverse=True):
        day = latest_served_date(horizon)
        if day is None:
            continue
        for slot, run in runs_on(day, roles_for(horizon).values()).items():
            if run.status in SERVING and slot not in found:
                found[slot] = (run, day)
    return [found[slot] for slot in PAGE_ORDER if slot in found]


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, DIGITS)


def recent_scores(models: list[tuple[ServingRun, date]]) -> dict[str, dict[str, Any]]:
    """model_monitor means over the six months up to each run's served day, in one query."""
    condition = Q()
    for run, day in models:
        start = months_before(day, RECENT_MONTHS)
        condition |= Q(run_id=run.run_id, horizon=run.horizon, date__gt=start, date__lte=day)
    rows = (
        ModelMonitor.objects.filter(condition)
        .values("run_id")
        .annotate(
            days=Count("date"),
            ic_sd=StdDev("ic", sample=True),
            **{name: Avg(name) for name in MONITOR_MEANS},
        )
    )
    found: dict[str, dict[str, Any]] = {row["run_id"]: dict(row) for row in rows}
    scores: dict[str, dict[str, Any]] = {}
    for run, day in models:
        row: dict[str, Any] = found.get(run.run_id, {})
        ic, spread = row.get("ic"), row.get("ic_sd")
        scores[run.run_id] = {
            "start": months_before(day, RECENT_MONTHS) + timedelta(days=1),
            "end": day,
            "days": row.get("days", 0),
            **{name: rounded(row.get(name)) for name in MONITOR_MEANS},
            "icir": rounded(ic / spread) if ic is not None and spread else None,
        }
    return scores


def research_totals(models: list[tuple[ServingRun, date]]) -> dict[tuple[str, int, str], Any]:
    """Research-period backtest totals of the rank runs and their ridge baselines."""
    ranks = [run for run, _ in models if run.objective == ModelObjective.RANK]
    if not ranks:
        return {}
    condition = Q()
    for run in ranks:
        condition |= Q(
            run_key=run.run_key,
            horizon=run.horizon,
            model_type__in=[run.model_type, BASELINE_MODEL],
        )
    rows = BacktestMetric.objects.filter(
        condition, objective=ModelObjective.RANK, grain=MetricGrain.TOTAL, metric__in=RANK_METRICS
    ).values(
        "run_key",
        "horizon",
        "model_type",
        "metric",
        "segment",
        "value",
        "period_start",
        "period_end",
    )
    totals: dict[tuple[str, int, str], Any] = {}
    for row in rows:
        key = (row["run_key"], row["horizon"], row["model_type"])
        entry = totals.setdefault(
            key,
            {
                "start": row["period_start"],
                "end": row["period_end"],
                "metrics": {},
                "liquidity": {},
            },
        )
        if row["segment"]:
            entry["liquidity"].setdefault(row["segment"], {})[row["metric"]] = row["value"]
        else:
            entry["metrics"][row["metric"]] = row["value"]
    return totals


def model_entry(
    run: ServingRun, day: date, recent: dict[str, Any], totals: dict[tuple[str, int, str], Any]
) -> dict[str, Any]:
    baseline = totals.get((run.run_key, run.horizon, BASELINE_MODEL))
    if run.model_type == BASELINE_MODEL:
        baseline = None
    return {
        "objective": run.objective,
        "horizon": run.horizon,
        **{name: getattr(run, name) for name in ENTRY_FIELDS},
        "created_at": run.created_at,
        "served_on": day,
        "recent": recent,
        "research": totals.get((run.run_key, run.horizon, run.model_type)),
        "baseline": None
        if baseline is None
        else {"model_type": BASELINE_MODEL, "metrics": baseline["metrics"]},
    }


class ModelListView(ReadOnlyView):
    def get(self, request: Request) -> Response:
        models = serving_models()
        recent = recent_scores(models) if models else {}
        totals = research_totals(models)
        return envelope([model_entry(run, day, recent[run.run_id], totals) for run, day in models])


def chart_source(horizon: int) -> tuple[str | None, str | None, str | None]:
    """The serving rank run's key, or before any serves, the latest researched one."""
    day = latest_served_date(horizon)
    slot = Slot(ModelObjective.RANK, horizon)
    run = None if day is None else runs_on(day, [slot]).get(slot)
    if run is not None:
        return run.run_key, run.model_type, "serving"
    latest = (
        BacktestMetric.objects.filter(
            objective=ModelObjective.RANK, horizon=horizon, grain=MetricGrain.MONTH, metric=MONTH_IC
        )
        .exclude(model_type=BASELINE_MODEL)
        .order_by("-created_at", "-id")
        .values("run_key", "model_type")
        .first()
    )
    if latest is None:
        return None, None, None
    return latest["run_key"], latest["model_type"], "research"


def monthly_rows(run_key: str, model_type: str, horizon: int) -> list[dict[str, Any]]:
    rows = BacktestMetric.objects.filter(
        run_key=run_key,
        objective=ModelObjective.RANK,
        horizon=horizon,
        grain=MetricGrain.MONTH,
        metric__in=[MONTH_IC, MONTH_DAYS],
        model_type__in=[model_type, BASELINE_MODEL],
    ).values("model_type", "metric", "period_start", "period_end", "value")
    by_month: dict[date, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        entry = by_month[row["period_start"]]
        entry["end"] = row["period_end"]
        side = "model" if row["model_type"] == model_type else "baseline"
        if row["metric"] == MONTH_DAYS:
            if side == "model":
                entry["days"] = None if row["value"] is None else int(row["value"])
        else:
            entry[side] = row["value"]
    full = settings.API_FULL_MONTH_DAYS
    return [
        {
            "month": start.strftime("%Y-%m"),
            "start": start,
            "end": entry["end"],
            "model": entry.get("model"),
            "baseline": entry.get("baseline"),
            "days": entry.get("days"),
            "complete": entry.get("days") is not None and entry["days"] >= full,
        }
        for start, entry in sorted(by_month.items())
        if "model" in entry
    ]


class MonthlyIcView(ReadOnlyView):
    def get(self, request: Request) -> Response:
        query = parse(MonthlyIcQuery, request)
        horizon = query["horizon"]
        run_key, model_type, source = chart_source(horizon)
        data = (
            []
            if run_key is None or model_type is None
            else monthly_rows(run_key, model_type, horizon)
        )
        data = data[-query["months"] :]
        meta = {
            "horizon": horizon,
            "run_key": run_key,
            "model_type": model_type,
            "baseline": BASELINE_MODEL,
            "source": source,
            "min_days": settings.API_FULL_MONTH_DAYS,
            "above_baseline": sum(
                row["model"] is not None
                and row["baseline"] is not None
                and row["model"] > row["baseline"]
                for row in data
            ),
            "negative": sum(row["model"] is not None and row["model"] < 0 for row in data),
        }
        return envelope(data, meta)
