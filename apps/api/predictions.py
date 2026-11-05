"""The daily prediction table: ranks merged with return, interval, volatility and probability."""

from collections import defaultdict
from datetime import date
from typing import Any

from django.db.models import F, OuterRef, Q, QuerySet, Subquery
from rest_framework.request import Request
from rest_framework.response import Response

from apps.api.envelope import envelope
from apps.api.pagination import EnvelopePagination
from apps.api.params import PredictionQuery, parse
from apps.api.pricing import Interval, choose_interval, price_at
from apps.api.serving import (
    ServingRun,
    Slot,
    latest_served_date,
    ranks_on,
    recent_mean,
    roles_for,
    runs_on,
)
from apps.api.views import ReadOnlyView
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import DailyPrice, StockMaster
from apps.research.models import FeatureDaily, ModelStatus, Prediction

PRICE_DIGITS = 4
RATIO_DIGITS = 6
BASE_FIELDS = ("stock_id", "rank", "pct", "score", "in_universe", "close", "close_date")
PART_FIELDS = ("run_id", "stock_id", "ret_pred", "q10", "q50", "q90", "vol_pred", "prob_up_5pct")


def rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def model_meta(role: str, run: ServingRun) -> dict[str, Any]:
    return {
        "role": role,
        "run_id": run.run_id,
        "objective": run.objective,
        "horizon": run.horizon,
        "model_type": run.model_type,
        "version": run.version,
        "status": run.status,
        "train_start": run.train_start,
        "train_end": run.train_end,
    }


def interval_json(interval: Interval | None) -> dict[str, Any] | None:
    if interval is None:
        return None
    return {
        "low": round(interval.low, PRICE_DIGITS),
        "high": round(interval.high, PRICE_DIGITS),
        "ret_low": round(interval.ret_low, RATIO_DIGITS),
        "ret_high": round(interval.ret_high, RATIO_DIGITS),
    }


def filtered(rows: QuerySet[Prediction], query: dict[str, Any]) -> QuerySet[Prediction]:
    if market := query.get("market"):
        rows = rows.filter(
            stock_id__in=StockMaster.objects.filter(market=market).values("stock_id")
        )
    if text := query.get("q"):
        named = StockMaster.objects.filter(name__icontains=text).values("stock_id")
        rows = rows.filter(Q(stock_id__istartswith=text) | Q(stock_id__in=named))
    return rows


def with_closes(rows: QuerySet[Prediction], day: date) -> QuerySet[Prediction]:
    """Each stock's last traded close on or before the day, looked up per row by index."""
    prices = DailyPrice.objects.filter(
        stock_id=OuterRef("stock_id"), date__lte=day, close__isnull=False
    ).order_by("-date")
    return rows.annotate(
        close=Subquery(prices.values("close")[:1]),
        close_date=Subquery(prices.values("date")[:1]),
    )


def parts_of(
    day: date, roles: dict[str, Slot], runs: dict[Slot, ServingRun], stock_ids: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """stock → role → that role's prediction row, for every role but the rank."""
    role_runs = {
        role: runs[slot] for role, slot in roles.items() if role != "rank" and slot in runs
    }
    if not role_runs or not stock_ids:
        return {}
    condition = Q()
    for run in role_runs.values():
        condition |= Q(run_id=run.run_id, horizon=run.horizon)
    rows = Prediction.objects.filter(condition, date=day, stock_id__in=stock_ids).values(
        *PART_FIELDS
    )
    by_run: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_run[row["run_id"]][row["stock_id"]] = row
    parts: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for role, run in role_runs.items():
        for stock_id, row in by_run[run.run_id].items():
            parts[stock_id][role] = row
    return parts


def previous_ranks(day: date, horizon: int, stock_ids: list[str]) -> dict[str, int | None]:
    """Ranks on the session before the day; none when the calendar cannot tell."""
    try:
        previous = TradingDays.from_db().shift(day, -1)
    except CalendarError:
        return {}
    return ranks_on(previous, horizon, stock_ids)


def needs_tech_vol(
    base: dict[str, Any], parts: dict[str, dict[str, Any]], horizon: int, quantile_alert: bool
) -> bool:
    """A priced row whose band can come neither from quantiles nor from vol_pred."""
    if base["close"] is None:
        return False
    band = parts.get("interval", {})
    interval = choose_interval(
        float(base["close"]),
        horizon,
        ret=None,
        q10=band.get("q10"),
        q90=band.get("q90"),
        vol=parts.get("volatility", {}).get("vol_pred"),
        tech_vol=None,
        quantile_alert=quantile_alert,
    )
    return interval is None


def tech_vols(day: date, stock_ids: list[str]) -> dict[str, float | None]:
    if not stock_ids:
        return {}
    rows = FeatureDaily.objects.filter(date=day, stock_id__in=stock_ids)
    return dict(rows.values_list("stock_id", "tech_vol_20d"))


def prediction_row(
    base: dict[str, Any],
    parts: dict[str, dict[str, Any]],
    master: dict[str, Any] | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    ret_row, band_row = parts.get("return", {}), parts.get("interval", {})
    ret = ret_row.get("ret_pred") if ret_row.get("ret_pred") is not None else ret_row.get("q50")
    vol = parts.get("volatility", {}).get("vol_pred")
    close = None if base["close"] is None else float(base["close"])
    interval = None
    if close is not None:
        interval = choose_interval(
            close,
            context["horizon"],
            ret=ret,
            q10=band_row.get("q10"),
            q90=band_row.get("q90"),
            vol=vol,
            tech_vol=context["tech_vols"].get(base["stock_id"]),
            quantile_alert=context["quantile_alert"],
        )
    return {
        "stock_id": base["stock_id"],
        "name": None if master is None else master["name"],
        "market": None if master is None else master["market"],
        "market_label": None if master is None else master["market_label"],
        "rank": base["rank"],
        "pct": base["pct"],
        "score": base["score"],
        "prev_rank": context["prev_ranks"].get(base["stock_id"]),
        "in_universe": base["in_universe"],
        "close": close,
        "close_date": base["close_date"],
        "ret_pred": ret,
        "price_pred": None
        if close is None or ret is None
        else rounded(price_at(close, ret), PRICE_DIGITS),
        "interval": interval_json(interval),
        "interval_source": None if interval is None else interval.source,
        "prob_up_5pct": parts.get("probability", {}).get("prob_up_5pct"),
        "vol_pred": vol,
    }


def masters_of(stock_ids: list[str]) -> dict[str, dict[str, Any]]:
    stocks = StockMaster.objects.filter(stock_id__in=stock_ids)
    return {
        stock.stock_id: {
            "name": stock.name,
            "market": stock.market,
            "market_label": stock.get_market_display(),
        }
        for stock in stocks
    }


def rank_rows(
    day: date | None, horizon: int, roles: dict[str, Slot], runs: dict[Slot, ServingRun]
) -> QuerySet[Prediction]:
    """The serving rank run's rows of the day, which every merged row starts from."""
    rank_run = runs.get(roles["rank"])
    if day is None or rank_run is None:
        return Prediction.objects.none()
    return Prediction.objects.filter(run_id=rank_run.run_id, date=day, horizon=horizon)


def served_models(roles: dict[str, Slot], runs: dict[Slot, ServingRun]) -> list[dict[str, Any]]:
    return [model_meta(role, runs[slot]) for role, slot in roles.items() if slot in runs]


def degraded_roles(roles: dict[str, Slot], runs: dict[Slot, ServingRun]) -> list[str]:
    return [
        role
        for role, slot in roles.items()
        if slot in runs and runs[slot].status == ModelStatus.DEGRADED
    ]


def merged_rows(
    day: date | None,
    horizon: int,
    roles: dict[str, Slot],
    runs: dict[Slot, ServingRun],
    page: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank rows joined with every other role's row, priced on each stock's close."""
    if day is None or not page:
        return []
    stock_ids = [row["stock_id"] for row in page]
    band = runs.get(roles["interval"]) if "interval" in roles else None
    alert = band is not None and band.status == ModelStatus.DEGRADED
    parts = parts_of(day, roles, runs, stock_ids)
    lacking = [
        row["stock_id"]
        for row in page
        if needs_tech_vol(row, parts.get(row["stock_id"], {}), horizon, alert)
    ]
    context = {
        "horizon": horizon,
        "quantile_alert": alert,
        "prev_ranks": previous_ranks(day, horizon, stock_ids),
        "tech_vols": tech_vols(day, lacking),
    }
    masters = masters_of(stock_ids)
    return [
        prediction_row(row, parts.get(row["stock_id"], {}), masters.get(row["stock_id"]), context)
        for row in page
    ]


class PredictionListView(ReadOnlyView):
    def get(self, request: Request) -> Response:
        query = parse(PredictionQuery, request)
        horizon = query["horizon"]
        day = query.get("date") or latest_served_date(horizon)
        roles = roles_for(horizon)
        runs = {} if day is None else runs_on(day, roles.values())
        base = rank_rows(day, horizon, roles, runs)
        rows = with_closes(filtered(base, query), day or date.min).order_by(
            F("rank").asc(nulls_last=True), F("score").desc(nulls_last=True), "stock_id"
        )
        paginator = EnvelopePagination()
        page: list[dict[str, Any]] = (
            paginator.paginate_queryset(rows.values(*BASE_FIELDS), request, view=self) or []
        )
        meta = {
            "date": day,
            "horizon": horizon,
            "universe": base.filter(in_universe=True).count(),
            "models": served_models(roles, runs),
            "degraded": degraded_roles(roles, runs),
            "monitor": self.monitor(day, roles, runs),
            **paginator.meta(),
        }
        return envelope(merged_rows(day, horizon, roles, runs, page), meta)

    def monitor(
        self, day: date | None, roles: dict[str, Slot], runs: dict[Slot, ServingRun]
    ) -> dict[str, Any]:
        ic, ic_days = recent_mean(runs.get(roles["rank"]), day, "ic")
        band = runs.get(roles["interval"]) if "interval" in roles else None
        coverage, coverage_days = recent_mean(band, day, "coverage")
        return {
            "rank_ic": ic,
            "rank_ic_days": ic_days,
            "coverage": coverage,
            "coverage_days": coverage_days,
        }
