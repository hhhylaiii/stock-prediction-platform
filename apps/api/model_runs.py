"""The serving models with recent monitoring, research totals and the monthly rank IC chart."""

from collections import defaultdict
from datetime import date, timedelta
from statistics import fmean, stdev
from typing import Any

from django.conf import settings
from django.db.models import Q
from django.utils import timezone
from rest_framework.request import Request
from rest_framework.response import Response

from apps.api.envelope import envelope
from apps.api.params import MonthlyIcQuery, parse
from apps.api.serving import (
    MONITOR_FIELDS,
    SERVING_FIELDS,
    ServingRun,
    Slot,
    best_runs,
    eligible,
    latest_days,
    latest_served_date,
    months_before,
    runs_on,
    served_monitor,
    serving_run,
)
from apps.api.views import ReadOnlyView
from apps.research.eval_report import RANK_METRICS
from apps.research.holdout import PRODUCTION
from apps.research.models import (
    BacktestMetric,
    HoldoutRun,
    HoldoutScope,
    MetricGrain,
    ModelObjective,
    ModelRun,
    ModelStatus,
)
from apps.research.monitor import DEGRADED_WINDOW

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
BASELINE_MODEL = "ridge"
RESEARCH_KEY = "research_run_key"
MONTH_IC, MONTH_DAYS = RANK_METRICS[0], RANK_METRICS[-1]
ENTRY_FIELDS = ("run_id", "run_key", "model_type", "version", "status", "train_start", "train_end")
TOTAL_FIELDS = ("run_key", "model_type", "metric", "segment", "value", "period_start", "period_end")
WINDOW_FIELDS = {"own_rows": int, "days": int, "mostly_predecessors": bool}
HOLDOUT_RELEASE = "holdout_release"
RELEASE_TEXTS = ("status", "reason", "released_at")
RELEASE_LISTS = ("failed", "flagged")


def serving_models() -> list[tuple[ServingRun, date]]:
    """Per page slot its active or degraded run on the slot's own latest served day."""
    days = latest_days(PAGE_ORDER)
    if not days:
        return []
    condition = Q()
    for slot, day in days.items():
        condition |= Q(run__objective=slot.objective, horizon=slot.horizon, date=day)
    rows = eligible().filter(condition).values(*SERVING_FIELDS).distinct()
    best = best_runs((serving_run(row) for row in rows), days)
    return [(best[slot], days[slot]) for slot in PAGE_ORDER if slot in best]


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, DIGITS)


def summary(rows: list[dict[str, Any]], day: date) -> dict[str, Any]:
    """Means of each monitor column over the served days, with ICIR and the runs behind them."""
    columns = {
        name: [row[name] for row in rows if row[name] is not None] for name in MONITOR_FIELDS
    }
    ics = columns["ic"]
    spread = stdev(ics) if len(ics) > 1 else 0.0
    return {
        "start": months_before(day, RECENT_MONTHS) + timedelta(days=1),
        "end": day,
        "days": len(rows),
        **{name: rounded(fmean(values)) if values else None for name, values in columns.items()},
        "icir": rounded(fmean(ics) / spread) if spread else None,
        "run_ids": list(dict.fromkeys(row["run_id"] for row in rows)),
    }


def recent_scores(models: list[tuple[ServingRun, date]]) -> dict[Slot, dict[str, Any]]:
    windows = {
        Slot(run.objective, run.horizon): (months_before(day, RECENT_MONTHS), day)
        for run, day in models
    }
    served = served_monitor(windows)
    return {slot: summary(served.get(slot, []), window[1]) for slot, window in windows.items()}


def rank_holdouts() -> list[str]:
    """The latest rank holdout attempt's run_key; earlier attempts are superseded."""
    rows = HoldoutRun.objects.filter(scope=HoldoutScope.RANK).order_by("-attempt")
    return list(rows.values_list("run_key", flat=True)[:1])


KEY_ROWS: dict[str, Q] = {
    MetricGrain.TOTAL: Q(metric__in=RANK_METRICS),
    MetricGrain.MONTH: Q(metric=MONTH_IC, segment=""),
}


def metric_key(
    run: ServingRun | None, params: dict[str, Any], holdouts: list[str], horizon: int, grain: str
) -> tuple[str | None, str | None, str | None]:
    """(run_key, model_type, source) of the first key with rows: params, own key, holdout."""
    candidates = [(key, "holdout") for key in holdouts]
    if run is not None:
        candidates = [(run.run_key, "self"), *candidates]
        if isinstance(params.get(RESEARCH_KEY), str):
            candidates = [(params[RESEARCH_KEY], "params"), *candidates]
    rows = BacktestMetric.objects.filter(
        KEY_ROWS[grain],
        run_key__in=[key for key, _ in candidates],
        objective=ModelObjective.RANK,
        horizon=horizon,
        grain=grain,
    )
    rows = (
        rows.filter(model_type=run.model_type) if run else rows.exclude(model_type=BASELINE_MODEL)
    )
    found: dict[str, set[str]] = defaultdict(set)
    for key, model_type in rows.values_list("run_key", "model_type").distinct():
        found[key].add(model_type)
    for key, source in candidates:
        if key in found:
            if len(found[key]) > 1:
                return None, None, None
            return key, found[key].pop(), source
    return None, None, None


def research_totals(run_key: str, model_type: str, horizon: int) -> dict[str, Any]:
    """Research-period totals of one rank model and its ridge baseline."""
    rows = BacktestMetric.objects.filter(
        run_key=run_key,
        objective=ModelObjective.RANK,
        horizon=horizon,
        grain=MetricGrain.TOTAL,
        metric__in=RANK_METRICS,
        model_type__in=[model_type, BASELINE_MODEL],
    ).values(*TOTAL_FIELDS)
    totals: dict[str, Any] = {}
    for row in rows:
        entry = totals.setdefault(
            row["model_type"],
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


def rank_research(run: ServingRun, params: dict[str, Any], holdouts: list[str]) -> dict[str, Any]:
    key, model_type, source = metric_key(run, params, holdouts, run.horizon, MetricGrain.TOTAL)
    totals = (
        {} if key is None or model_type is None else research_totals(key, model_type, run.horizon)
    )
    baseline = None if model_type == BASELINE_MODEL else totals.get(BASELINE_MODEL)
    return {
        "metric_run_key": key,
        "metric_source": source,
        "research": None if model_type is None else totals.get(model_type),
        "baseline": None
        if baseline is None
        else {"model_type": BASELINE_MODEL, "metrics": baseline["metrics"]},
    }


NO_RESEARCH = {"metric_run_key": None, "metric_source": None, "research": None, "baseline": None}


def degraded_window(params: dict[str, Any]) -> dict[str, Any] | None:
    """params.degraded_window's three fields, or None when it is missing or unreadable."""
    window = params.get(DEGRADED_WINDOW)
    if not isinstance(window, dict):
        return None
    if any(type(window.get(name)) is not kind for name, kind in WINDOW_FIELDS.items()):
        return None
    return {name: window[name] for name in WINDOW_FIELDS}


def is_names(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(name, str) for name in value)


def holdout_release(params: dict[str, Any]) -> dict[str, Any] | None:
    """The released verdict a production run was built on, without its report path or hash."""
    production = params.get(PRODUCTION)
    release = production.get(HOLDOUT_RELEASE) if isinstance(production, dict) else None
    if not isinstance(release, dict):
        return None
    shown = {name: release.get(name) for name in (*RELEASE_TEXTS, *RELEASE_LISTS)}
    texts = all(isinstance(shown[name], str) for name in RELEASE_TEXTS)
    lists = all(is_names(shown[name]) for name in RELEASE_LISTS)
    return shown if texts and lists else None


class ModelListView(ReadOnlyView):
    def get(self, request: Request) -> Response:
        models = serving_models()
        recent = recent_scores(models)
        ranks = [run.run_id for run, _ in models if run.objective == ModelObjective.RANK]
        served = [run.run_id for run, _ in models]
        params = dict(ModelRun.objects.filter(run_id__in=served).values_list("run_id", "params"))
        holdouts = rank_holdouts() if ranks else []
        data = []
        for run, day in models:
            research = NO_RESEARCH
            if run.objective == ModelObjective.RANK:
                research = rank_research(run, params.get(run.run_id) or {}, holdouts)
            data.append(
                {
                    "objective": run.objective,
                    "horizon": run.horizon,
                    **{name: getattr(run, name) for name in ENTRY_FIELDS},
                    "created_at": timezone.localtime(run.created_at),
                    "served_on": day,
                    "degraded_on": run.degraded_on,
                    "degraded_window": (
                        degraded_window(params.get(run.run_id) or {})
                        if run.status == ModelStatus.DEGRADED
                        else None
                    ),
                    "holdout_release": holdout_release(params.get(run.run_id) or {}),
                    "recent": recent[Slot(run.objective, run.horizon)],
                    **research,
                }
            )
        return envelope(data)


def chart_source(horizon: int) -> tuple[str | None, str | None, str | None]:
    """The serving rank run's research key by params, its own key, then the rank holdout."""
    day = latest_served_date(horizon)
    slot = Slot(ModelObjective.RANK, horizon)
    run = None if day is None else runs_on(day, [slot]).get(slot)
    params: dict[str, Any] = {}
    if run is not None:
        params = ModelRun.objects.filter(run_id=run.run_id).values_list("params", flat=True)[0]
    return metric_key(run, params or {}, rank_holdouts(), horizon, MetricGrain.MONTH)


def monthly_rows(run_key: str, model_type: str, horizon: int) -> list[dict[str, Any]]:
    rows = BacktestMetric.objects.filter(
        run_key=run_key,
        objective=ModelObjective.RANK,
        horizon=horizon,
        grain=MetricGrain.MONTH,
        segment="",
        metric__in=[MONTH_IC, MONTH_DAYS],
        model_type__in=[model_type, BASELINE_MODEL],
    ).values("model_type", "metric", "period_start", "period_end", "value")
    by_month: dict[date, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        entry = by_month[row["period_start"]]
        entry["end"] = row["period_end"]
        side = "model" if row["model_type"] == model_type else "baseline"
        if row["metric"] == MONTH_IC:
            entry[side] = row["value"]
        elif side == "model":
            entry["days"] = None if row["value"] is None else int(row["value"])
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
        data = []
        if run_key is not None and model_type is not None:
            data = monthly_rows(run_key, model_type, horizon)[-query["months"] :]
        complete = [row for row in data if row["complete"]]
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
                for row in complete
            ),
            "negative": sum(row["model"] is not None and row["model"] < 0 for row in complete),
        }
        return envelope(data, meta)
