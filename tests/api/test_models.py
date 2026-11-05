from datetime import date, timedelta
from statistics import fmean, stdev
from typing import Any

import pytest
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.api.model_runs import MONTH_DAYS, MONTH_IC
from apps.research.eval_report import RANK_METRICS
from apps.research.models import BacktestMetric, ModelMonitor, ModelRun
from tests.api.conftest import (
    OCT_30,
    Runs,
    add_model_run,
    add_predictions,
    envelope_of,
    sessions_until,
    taipei,
)

pytestmark = pytest.mark.usefixtures("calendar_in_db")

RESEARCH = (date(2014, 1, 2), date(2024, 12, 3))


def add_metric(
    model_type: str,
    metric: str,
    value: float | None,
    grain: str = "total",
    period: tuple[date, date] = RESEARCH,
    segment: str = "",
    horizon: int = 20,
    run_key: str = "rank-serving",
) -> None:
    BacktestMetric.objects.create(
        run_key=run_key,
        model_type=model_type,
        objective="rank",
        horizon=horizon,
        metric=metric,
        grain=grain,
        period_start=period[0],
        period_end=period[1],
        segment=segment,
        value=value,
    )


def by_slot(body: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["objective"], row["horizon"]): row for row in body["data"]}


@pytest.fixture
def monitored(runs: Runs) -> Runs:
    days = sessions_until(OCT_30)
    ics = [0.02 + 0.01 * (n % 5) for n in range(len(days))]
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.rank20, date=day, horizon=20, ic=ic, n_stocks=1800)
        for day, ic in zip(days, ics, strict=True)
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.rank20, date=day, horizon=20, ic=0.9, n_stocks=1800)
        for day in (date(2026, 3, 2), date(2026, 4, 30))
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.quantile, date=day, horizon=20, coverage=0.79, width=0.13, n_stocks=9)
        for day in days[-30:]
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.prob, date=day, horizon=20, brier=0.231, auc=0.58, n_stocks=9)
        for day in days[-30:]
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.reg5, date=day, horizon=5, mae=0.058, direction_acc=0.54, n_stocks=9)
        for day in days[-30:]
    )
    return runs


def test_every_serving_model_is_listed_in_page_order(client: Client, runs: Runs) -> None:
    response = client.get("/api/models")

    assert response.status_code == 200
    body = envelope_of(response)
    assert [(row["objective"], row["horizon"]) for row in body["data"]] == [
        ("rank", 20),
        ("rank", 5),
        ("regression", 5),
        ("quantile", 20),
        ("volatility", 20),
        ("classification", 20),
    ]
    first = body["data"][0]
    assert {key: first[key] for key in ("run_id", "run_key", "model_type", "version")} == {
        "run_id": "rank-h20",
        "run_key": "rank-serving",
        "model_type": "lightgbm",
        "version": "2026-10",
    }
    assert (first["status"], first["train_start"], first["train_end"]) == (
        "active",
        "2023-10-01",
        "2026-09-30",
    )
    assert first["served_on"] == "2026-10-30"


def test_recent_scores_average_the_last_six_months_of_monitoring(
    client: Client, monitored: Runs
) -> None:
    rows = by_slot(envelope_of(client.get("/api/models")))
    days = sessions_until(OCT_30)
    ics = [0.02 + 0.01 * (n % 5) for n in range(len(days))]
    window = days

    rank = rows[("rank", 20)]["recent"]
    assert (rank["start"], rank["end"], rank["days"]) == ("2026-05-01", "2026-10-30", len(window))
    assert rank["ic"] == round(fmean(ics), 4)
    assert rank["icir"] == round(fmean(ics) / stdev(ics), 4)
    assert (
        rows[("quantile", 20)]["recent"]["coverage"],
        rows[("quantile", 20)]["recent"]["width"],
    ) == (
        0.79,
        0.13,
    )
    assert rows[("classification", 20)]["recent"]["auc"] == 0.58
    assert rows[("regression", 5)]["recent"]["mae"] == 0.058
    assert rows[("volatility", 20)]["recent"]["days"] == 0


def test_rank_models_carry_research_totals_and_the_ridge_baseline(
    client: Client, runs: Runs
) -> None:
    for metric, value in (("rank_ic", 0.041), ("rank_icir", 0.62), ("rank_ic_nw_t", 3.1)):
        add_metric("lightgbm", metric, value)
    add_metric("lightgbm", "beat_baseline_folds", None)
    add_metric("ridge", "rank_ic", 0.027)
    add_metric("ridge", "rank_icir", 0.41)
    add_metric("lightgbm", "rank_ic", 0.05, segment="liq_t3")
    add_metric("lightgbm", "rank_ic_nw_t", 2.4, segment="liq_t3")
    add_metric(
        "lightgbm",
        "positive_fold_share",
        0.9,
        grain="fold",
        period=(date(2020, 1, 2), date(2020, 6, 30)),
    )
    add_metric("lightgbm", "neutral_rank_ic", 0.8, run_key="rank-other")
    add_metric("lightgbm", "rank_ic", 0.7, horizon=5)

    rows = by_slot(envelope_of(client.get("/api/models")))

    research = rows[("rank", 20)]["research"]
    assert research == {
        "start": "2014-01-02",
        "end": "2024-12-03",
        "metrics": {
            "rank_ic": 0.041,
            "rank_icir": 0.62,
            "rank_ic_nw_t": 3.1,
            "beat_baseline_folds": None,
        },
        "liquidity": {"liq_t3": {"rank_ic": 0.05, "rank_ic_nw_t": 2.4}},
    }
    assert rows[("rank", 20)]["baseline"] == {
        "model_type": "ridge",
        "metrics": {"rank_ic": 0.027, "rank_icir": 0.41},
    }
    assert rows[("rank", 5)]["research"]["metrics"] == {"rank_ic": 0.7}
    assert rows[("rank", 5)]["baseline"] is None
    assert (rows[("quantile", 20)]["research"], rows[("quantile", 20)]["baseline"]) == (None, None)


def test_a_serving_ridge_model_is_not_its_own_baseline(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(model_type="ridge")
    add_metric("ridge", "rank_ic", 0.027)

    rank = by_slot(envelope_of(client.get("/api/models")))[("rank", 20)]

    assert rank["research"]["metrics"] == {"rank_ic": 0.027}
    assert rank["baseline"] is None


def test_candidates_screening_and_retired_runs_are_not_listed(client: Client, runs: Runs) -> None:
    rows = {"2330": {"score": 1.0, "rank": 1, "pct": 1.0}}
    for run_id, status, model_type in (
        ("rank-cand", "candidate", "lightgbm"),
        ("screening", "active", "screening"),
    ):
        add_predictions(
            add_model_run(run_id, "rank", status=status, model_type=model_type), OCT_30, 20, rows
        )
    ModelRun.objects.filter(pk="vol-h20").update(status="retired")

    body = envelope_of(client.get("/api/models"))

    assert [row["run_id"] for row in body["data"]] == [
        "rank-h20",
        "rank-h5",
        "reg-h5",
        "quantile-h20",
        "prob-h20",
    ]


def test_a_degraded_model_stays_listed(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")

    rows = by_slot(envelope_of(client.get("/api/models")))

    assert rows[("quantile", 20)]["status"] == "degraded"


def test_no_models_served_yet(client: Client, db: None) -> None:
    assert envelope_of(client.get("/api/models"))["data"] == []


def test_models_query_count(
    client: Client, monitored: Runs, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    add_metric("lightgbm", "rank_ic", 0.041)
    add_metric("ridge", "rank_ic", 0.027)

    with django_assert_max_num_queries(7):
        client.get("/api/models")


def test_monthly_metric_names_come_from_the_rank_metric_list() -> None:
    assert (MONTH_IC, MONTH_DAYS) == ("rank_ic", "rank_ic_days")
    assert {MONTH_IC, MONTH_DAYS} <= set(RANK_METRICS)


def month_starts(first: date, count: int) -> list[date]:
    months, year, month = [], first.year, first.month
    for _ in range(count):
        months.append(date(year, month, 1))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def month_end(start: date) -> date:
    following = month_starts(start, 2)[1]
    return following - timedelta(days=1)


MONTHS = month_starts(date(2022, 11, 1), 26)


def add_months(model_type: str, horizon: int = 20, run_key: str = "rank-serving") -> None:
    for n, start in enumerate(MONTHS):
        period = (start, month_end(start))
        value = (0.03 if model_type == "lightgbm" else 0.02) + (-0.06 if n in (4, 9) else 0.0)
        add_metric(model_type, "rank_ic", value, "month", period, horizon=horizon, run_key=run_key)
        days = 2.0 if start == MONTHS[-1] else 21.0
        add_metric(
            model_type, "rank_ic_days", days, "month", period, horizon=horizon, run_key=run_key
        )


def test_monthly_ic_returns_the_last_24_months_against_ridge(client: Client, runs: Runs) -> None:
    add_months("lightgbm")
    add_months("ridge")

    body = envelope_of(client.get("/api/models/ic-monthly"))

    data = body["data"]
    assert [row["month"] for row in data] == [start.strftime("%Y-%m") for start in MONTHS[-24:]]
    assert data[0] == {
        "month": "2023-01",
        "start": "2023-01-01",
        "end": "2023-01-31",
        "model": 0.03,
        "baseline": 0.02,
        "days": 21,
        "complete": True,
    }
    assert (data[-1]["days"], data[-1]["complete"]) == (2, False)
    assert body["meta"] == {
        "horizon": 20,
        "run_key": "rank-serving",
        "model_type": "lightgbm",
        "baseline": "ridge",
        "source": "serving",
        "min_days": 10,
        "above_baseline": 24,
        "negative": 2,
    }


def test_monthly_ic_for_five_days_reads_the_five_day_rows(client: Client, runs: Runs) -> None:
    add_months("lightgbm")
    add_months("lightgbm", horizon=5)
    BacktestMetric.objects.filter(horizon=5, metric="rank_ic").update(value=0.01)

    data = envelope_of(client.get("/api/models/ic-monthly", {"horizon": "5", "months": "3"}))[
        "data"
    ]

    assert [row["model"] for row in data] == [0.01, 0.01, 0.01]
    assert data[0]["baseline"] is None


def test_before_any_model_serves_the_latest_research_run_is_shown(client: Client, db: None) -> None:
    add_months("lightgbm", run_key="rank-old")
    add_months("lightgbm", run_key="rank-new")
    BacktestMetric.objects.filter(run_key="rank-new").update(created_at=taipei(2030, 1, 1))
    BacktestMetric.objects.filter(run_key="rank-new", metric="rank_ic").update(value=0.05)

    body = envelope_of(client.get("/api/models/ic-monthly", {"months": "2"}))

    assert [row["model"] for row in body["data"]] == [0.05, 0.05]
    assert (body["meta"]["run_key"], body["meta"]["source"]) == ("rank-new", "research")


def test_no_monthly_metrics_is_an_empty_chart(client: Client, db: None) -> None:
    body = envelope_of(client.get("/api/models/ic-monthly"))

    assert body["data"] == []
    assert (body["meta"]["run_key"], body["meta"]["source"]) == (None, None)


@pytest.mark.parametrize(
    ("params", "field"),
    [({"horizon": "10"}, "horizon"), ({"months": "0"}, "months"), ({"months": "121"}, "months")],
)
def test_bad_monthly_queries_are_invalid(
    client: Client, db: None, params: dict[str, str], field: str
) -> None:
    response = client.get("/api/models/ic-monthly", params)

    assert response.status_code == 400
    assert set(envelope_of(response)["error"]["fields"]) == {field}


def test_monthly_ic_query_count(
    client: Client, runs: Runs, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    add_months("lightgbm")
    add_months("ridge")

    with django_assert_max_num_queries(4):
        client.get("/api/models/ic-monthly")
