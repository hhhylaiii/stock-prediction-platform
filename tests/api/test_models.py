import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import pytest
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.api import serving
from apps.api.model_runs import MONTH_DAYS, MONTH_IC
from apps.api.serving import HOLDOUT_RELEASE
from apps.etl.models import DataQualityLog
from apps.research import production, promotion
from apps.research.eval_report import RANK_METRICS
from apps.research.holdout import (
    FAILED,
    claim_holdout,
    production_pass,
    release_holdout,
    released_verdict,
)
from apps.research.models import BacktestMetric, HoldoutRun, ModelMonitor, ModelRun, Prediction
from tests.api.conftest import (
    EARLY_CALENDAR_START,
    FORCED_REASON,
    OCT_29,
    OCT_30,
    Runs,
    add_model_run,
    add_predictions,
    envelope_of,
    forced_params,
    sessions_until,
    taipei,
    trained_until,
)
from tests.research.run_panels import write_holdout_report

pytestmark = pytest.mark.usefixtures("calendar_in_db")


@pytest.fixture
def calendar_start() -> date:
    """Runs here train before July, so their out-of-sample days start inside the calendar."""
    return EARLY_CALENDAR_START


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


SWITCH = date(2026, 9, 30)


def ic_on(n: int) -> float:
    return 0.02 + 0.01 * (n % 5)


@pytest.fixture
def monitored(runs: Runs) -> Runs:
    """The rank line: an old run serves until the switch, the current one after it."""
    days = sessions_until(OCT_30)
    ModelRun.objects.filter(pk="rank-h20").update(train_end=trained_until(SWITCH))
    old = add_model_run("rank-old", "rank", status="retired", train_end=date(2026, 1, 30))
    ModelMonitor.objects.bulk_create(
        ModelMonitor(
            run=old if day <= SWITCH else runs.rank20,
            date=day,
            horizon=20,
            ic=ic_on(n),
            n_stocks=1800,
        )
        for n, day in enumerate(days)
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.rank20, date=day, horizon=20, ic=0.9, n_stocks=1800)
        for day in days
        if day <= SWITCH
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=old, date=day, horizon=20, ic=0.9, n_stocks=1800)
        for day in (date(2026, 3, 2), date(2026, 4, 30), *[day for day in days if day > SWITCH])
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
    assert first["created_at"] == "2026-10-01T00:00:00+08:00"


def test_recent_scores_average_the_last_six_months_of_monitoring(
    client: Client, monitored: Runs
) -> None:
    rows = by_slot(envelope_of(client.get("/api/models")))
    days = sessions_until(OCT_30)
    ics = [ic_on(n) for n in range(len(days))]

    rank = rows[("rank", 20)]["recent"]
    assert (rank["start"], rank["end"], rank["days"]) == ("2026-05-01", "2026-10-30", len(days))
    assert rank["ic"] == round(fmean(ics), 4)
    assert rank["icir"] == round(fmean(ics) / stdev(ics), 4)
    assert rank["run_ids"] == ["rank-old", "rank-h20"]
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


def test_recent_scores_skip_days_inside_a_run_s_training_window(client: Client, runs: Runs) -> None:
    days = sessions_until(OCT_30)
    ModelRun.objects.filter(pk="rank-h20").update(train_end=trained_until(SWITCH))
    ModelMonitor.objects.bulk_create(
        ModelMonitor(
            run=runs.rank20, date=day, horizon=20, ic=0.5 if day <= SWITCH else 0.01, n_stocks=9
        )
        for day in days
    )
    for run_id, status, model_type in (
        ("rank-cand", "candidate", "lightgbm"),
        ("screening", "active", "screening"),
    ):
        other = add_model_run(
            run_id, "rank", status=status, model_type=model_type, train_end=date(2026, 1, 30)
        )
        ModelMonitor.objects.bulk_create(
            ModelMonitor(run=other, date=day, horizon=20, ic=0.9, n_stocks=9)
            for day in days
            if day <= SWITCH
        )

    rank = by_slot(envelope_of(client.get("/api/models")))[("rank", 20)]["recent"]

    assert (rank["days"], rank["ic"]) == (len([day for day in days if day > SWITCH]), 0.01)


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
    add_metric("lightgbm", "decile_excess", 0.3)
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
    assert (rows[("rank", 20)]["metric_run_key"], rows[("rank", 20)]["metric_source"]) == (
        "rank-serving",
        "self",
    )
    assert rows[("rank", 5)]["research"]["metrics"] == {"rank_ic": 0.7}
    assert rows[("rank", 5)]["baseline"] is None
    assert (rows[("quantile", 20)]["research"], rows[("quantile", 20)]["baseline"]) == (None, None)
    assert rows[("quantile", 20)]["metric_source"] is None


def test_the_run_s_params_name_its_research_run_key(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(params={"research_run_key": "rank-research"})
    add_metric("lightgbm", "rank_ic", 0.8)
    add_metric("lightgbm", "rank_ic", 0.041, run_key="rank-research")
    add_metric("ridge", "rank_ic", 0.027, run_key="rank-research")

    rank = by_slot(envelope_of(client.get("/api/models")))[("rank", 20)]

    assert rank["research"]["metrics"] == {"rank_ic": 0.041}
    assert rank["baseline"]["metrics"] == {"rank_ic": 0.027}
    assert (rank["metric_run_key"], rank["metric_source"]) == ("rank-research", "params")


def test_without_own_metrics_the_latest_rank_holdout_is_read(client: Client, runs: Runs) -> None:
    for attempt, run_key in ((1, "rank-frozen"), (2, "rank-refrozen")):
        HoldoutRun.objects.create(
            scope="rank", attempt=attempt, run_key=run_key, last_date=OCT_30, reason="again"
        )
        add_metric("lightgbm", "rank_ic", 0.01 * attempt, run_key=run_key)
    HoldoutRun.objects.create(scope="return", attempt=1, run_key="return-frozen", last_date=OCT_30)
    add_metric("lightgbm", "rank_ic", 0.9, run_key="return-frozen")

    rank = by_slot(envelope_of(client.get("/api/models")))[("rank", 20)]

    assert rank["research"]["metrics"] == {"rank_ic": 0.02}
    assert (rank["metric_run_key"], rank["metric_source"]) == ("rank-refrozen", "holdout")


def test_a_superseded_holdout_is_not_read(client: Client, runs: Runs) -> None:
    for attempt, run_key in ((1, "rank-frozen"), (2, "rank-refrozen")):
        HoldoutRun.objects.create(
            scope="rank", attempt=attempt, run_key=run_key, last_date=OCT_30, reason="again"
        )
    HoldoutRun.objects.filter(run_key="rank-frozen").update(claimed_at=taipei(2030, 1, 1))
    add_metric("lightgbm", "rank_ic", 0.01, run_key="rank-frozen")

    rank = by_slot(envelope_of(client.get("/api/models")))[("rank", 20)]

    assert (rank["metric_run_key"], rank["metric_source"], rank["research"]) == (None, None, None)


def test_a_key_without_rank_metrics_is_skipped(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(params={"research_run_key": "rank-research"})
    add_metric("lightgbm", "decile_excess", 0.3, run_key="rank-research")
    add_metric("lightgbm", "rank_ic", 0.041)
    add_metric(
        "lightgbm",
        "rank_ic",
        0.5,
        "month",
        (MONTHS[0], month_end(MONTHS[0])),
        segment="liq_t1",
        run_key="rank-research",
    )
    add_months("lightgbm")

    rank = by_slot(envelope_of(client.get("/api/models")))[("rank", 20)]
    chart = envelope_of(client.get("/api/models/ic-monthly"))

    assert (rank["metric_run_key"], rank["research"]["metrics"]) == (
        "rank-serving",
        {"rank_ic": 0.041},
    )
    assert (chart["meta"]["run_key"], chart["meta"]["source"]) == ("rank-serving", "self")


def test_a_non_rank_run_sharing_a_run_key_gets_no_rank_research(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(run_key="rank-serving")
    add_metric("lightgbm", "rank_ic", 0.041)

    quantile = by_slot(envelope_of(client.get("/api/models")))[("quantile", 20)]

    assert (quantile["research"], quantile["baseline"]) == (None, None)


def test_a_serving_ridge_model_is_not_its_own_baseline(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(model_type="ridge")
    add_metric("ridge", "rank_ic", 0.027)
    add_metric("lightgbm", "rank_ic", 0.5)

    rank = by_slot(envelope_of(client.get("/api/models")))[("rank", 20)]

    assert rank["research"]["metrics"] == {"rank_ic": 0.027}
    assert rank["baseline"] is None


def test_candidates_screening_and_retired_runs_are_not_listed(client: Client, runs: Runs) -> None:
    rows = {"2330": {"score": 1.0, "rank": 1, "pct": 1.0}}
    for run_id, status, model_type in (
        ("rank-cand", "candidate", "lightgbm"),
        ("screening", "active", "screening"),
    ):
        newer = add_model_run(
            run_id, "rank", status=status, model_type=model_type, created_at=taipei(2026, 10, 20)
        )
        add_predictions(newer, OCT_30, 20, rows)
    ModelRun.objects.filter(pk="vol-h20").update(status="retired")

    body = envelope_of(client.get("/api/models"))

    assert [row["run_id"] for row in body["data"]] == [
        "rank-h20",
        "rank-h5",
        "reg-h5",
        "quantile-h20",
        "prob-h20",
    ]


def test_each_slot_follows_its_own_latest_served_day(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk="rank-h5").update(status="retired")

    rows = by_slot(envelope_of(client.get("/api/models")))

    assert ("rank", 5) not in rows
    assert rows[("regression", 5)]["run_id"] == "reg-h5"


def test_a_slot_that_stopped_earlier_shows_its_own_last_day(client: Client, runs: Runs) -> None:
    Prediction.objects.filter(run=runs.vol, date=OCT_30).delete()
    add_predictions(runs.vol, OCT_29, 20, {"2330": {"vol_pred": 0.3}})

    vol = by_slot(envelope_of(client.get("/api/models")))[("volatility", 20)]

    assert vol["served_on"] == "2026-10-29"


def test_a_degraded_model_stays_listed(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")

    rows = by_slot(envelope_of(client.get("/api/models")))

    assert rows[("quantile", 20)]["status"] == "degraded"


WINDOW = {"own_rows": 20, "days": 60, "mostly_predecessors": True}


def test_a_degraded_model_carries_its_day_and_window_but_no_other_params(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(
        status="degraded",
        params={
            "degraded_on": "2026-10-20",
            "degraded_window": {**WINDOW, "note": "kept out"},
            "gates": {"agreement": 0.9},
            "research_run_key": "interval-research",
        },
    )

    rows = by_slot(envelope_of(client.get("/api/models")))

    quantile, rank = rows[("quantile", 20)], rows[("rank", 20)]
    assert (quantile["degraded_on"], quantile["degraded_window"]) == ("2026-10-20", WINDOW)
    assert not {"params", "gates", "research_run_key", "note"} & set(quantile)
    assert (rank["degraded_on"], rank["degraded_window"]) == (None, None)


@pytest.mark.parametrize(
    "window",
    [
        "40/60",
        {"own_rows": 40, "days": 60},
        {**WINDOW, "own_rows": True},
        {**WINDOW, "days": "60"},
        {**WINDOW, "mostly_predecessors": 1},
    ],
)
def test_an_unreadable_degraded_window_is_left_out(
    client: Client, runs: Runs, window: object
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(
        status="degraded", params={"degraded_on": "2026-10-20", "degraded_window": window}
    )

    rows = by_slot(envelope_of(client.get("/api/models")))

    assert rows[("quantile", 20)]["degraded_window"] is None


def test_a_window_left_behind_on_a_restored_run_is_not_served(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(
        status="active", params={"degraded_window": WINDOW}
    )

    rows = by_slot(envelope_of(client.get("/api/models")))

    assert (rows[("quantile", 20)]["degraded_on"], rows[("quantile", 20)]["degraded_window"]) == (
        None,
        None,
    )


REPORT_PATH = "/srv/host/reports/rank-holdout-20261111.md"
RELEASE: dict[str, Any] = {
    "status": "未達標",
    "failed": ["排名 20 日 IC"],
    "flagged": ["排名 5 日 IC"],
    "reason": "差距在 seed 間標準差內，先上線觀察",
    "released_at": "2026-11-12T10:03:00+08:00",
    "report": REPORT_PATH,
    "sha256": "ab" * 32,
}
RELEASE_SHOWN = ("status", "failed", "flagged", "reason", "released_at")


def released(release: object) -> dict[str, Any]:
    return {"production": {"role": "model", "frozen_by": "rank-holdout", HOLDOUT_RELEASE: release}}


def test_the_release_key_is_the_one_a_production_run_records() -> None:
    assert HOLDOUT_RELEASE == production.HOLDOUT_RELEASE


def test_a_model_built_on_a_released_holdout_carries_the_release_without_its_report(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(params=released(RELEASE))

    response = client.get("/api/models")

    rows = by_slot(envelope_of(response))
    assert rows[("rank", 20)]["holdout_release"] == {name: RELEASE[name] for name in RELEASE_SHOWN}
    assert rows[("quantile", 20)]["holdout_release"] is None
    body = response.content.decode()
    assert REPORT_PATH not in body
    assert RELEASE["sha256"] not in body


def test_the_release_a_build_records_reaches_the_endpoint_with_its_reason(
    client: Client, runs: Runs, settings: Any, tmp_path: Path
) -> None:
    settings.RESEARCH_CACHE_DIR = str(tmp_path)
    claim_holdout("rank", "rank-final", OCT_30)
    write_holdout_report("rank", "rank-final", FAILED, REPORT_PATH)
    release_holdout("rank", FAILED, "the user accepts the gap")
    recorded = released_verdict(production_pass("rank"))
    assert recorded is not None
    ModelRun.objects.filter(pk="rank-h20").update(params=released(recorded))

    response = client.get("/api/models")

    shown = by_slot(envelope_of(response))[("rank", 20)]["holdout_release"]
    assert shown == {name: recorded[name] for name in RELEASE_SHOWN}
    assert (shown["status"], shown["reason"]) == (FAILED, "the user accepts the gap")
    assert shown["failed"]
    assert shown["flagged"]
    assert REPORT_PATH not in response.content.decode()
    served = envelope_of(client.get("/api/predictions"))["meta"]["models"]
    assert [model["released_verdict"] for model in served if model["role"] == "rank"] == [FAILED]


@pytest.mark.parametrize(
    "release",
    [
        "未達標",
        {name: value for name, value in RELEASE.items() if name != "reason"},
        {**RELEASE, "status": None},
        {**RELEASE, "status": "達標"},
        {**RELEASE, "released_at": 20261112},
        {**RELEASE, "failed": "排名 20 日 IC"},
        {**RELEASE, "flagged": [1]},
    ],
)
def test_an_unreadable_holdout_release_is_left_out(
    client: Client, runs: Runs, release: object
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(params=released(release))

    rows = by_slot(envelope_of(client.get("/api/models")))

    assert rows[("rank", 20)]["holdout_release"] is None


def test_production_params_that_are_not_a_mapping_carry_no_release(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(params={"production": "rank-2026-11"})

    rows = by_slot(envelope_of(client.get("/api/models")))

    assert rows[("rank", 20)]["holdout_release"] is None


def test_the_promotion_key_is_the_one_promote_records() -> None:
    assert serving.PROMOTION == promotion.PROMOTION


def test_a_model_swapped_in_by_force_carries_the_day_the_reason_and_the_failed_gates(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(params=forced_params("rank-h20"))

    rows = by_slot(envelope_of(client.get("/api/models")))

    assert rows[("rank", 20)]["forced_promotion"] == {
        "promoted_on": "2026-10-30",
        "reason": FORCED_REASON,
        "gates": [
            {
                "name": "agreement",
                "reason": "",
                "value": 0.412,
                "threshold": ">= 0.5",
                "sample": 1842,
                "sample_unit": "stocks",
            },
            {
                "name": "shadow_oos",
                "reason": "no shadow rows",
                "value": -0.004,
                "threshold": "> 0",
                "sample": 120,
                "sample_unit": "days",
            },
        ],
    }
    assert rows[("quantile", 20)]["forced_promotion"] is None


def test_a_forced_promotion_serves_only_the_listed_gate_fields_and_a_text_reason(
    client: Client, runs: Runs
) -> None:
    params = forced_params("rank-h20")
    params[promotion.PROMOTION]["reason"] = ["not", "text"]
    stored = params[promotion.PROMOTION]["failed"][0]["gates"][0]
    stored["errors"] = [{"date": "2026-10-01", "check": "/srv/host/errors.log"}]
    ModelRun.objects.filter(pk="rank-h20").update(params=params)

    response = client.get("/api/models")

    forced = by_slot(envelope_of(response))[("rank", 20)]["forced_promotion"]
    assert forced["reason"] is None
    assert [sorted(gate) for gate in forced["gates"]] == [sorted(serving.GATE_FIELDS)] * 2
    assert "/srv/host/errors.log" not in response.content.decode()


def test_no_models_served_yet(client: Client, db: None) -> None:
    assert envelope_of(client.get("/api/models"))["data"] == []


def test_models_query_count(
    client: Client, monitored: Runs, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    add_metric("lightgbm", "rank_ic", 0.041)
    add_metric("ridge", "rank_ic", 0.027)
    for run, horizon, row in (
        (monitored.vol, 20, {"vol_pred": 0.3}),
        (monitored.prob, 20, {"prob_up_5pct": 0.6}),
        (monitored.reg5, 5, {"ret_pred": 0.01}),
    ):
        Prediction.objects.filter(run=run, date=OCT_30).delete()
        add_predictions(run, OCT_29, horizon, {"2330": row})

    with django_assert_max_num_queries(10):
        body = envelope_of(client.get("/api/models"))

    assert len(body["data"]) == 6


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
    """26 months; two negative ones, a 10-day month and a 2-day last month."""
    for n, start in enumerate(MONTHS):
        period = (start, month_end(start))
        value = (0.03 if model_type == "lightgbm" else 0.02) + (-0.06 if n in (4, 9) else 0.0)
        add_metric(model_type, "rank_ic", value, "month", period, horizon=horizon, run_key=run_key)
        days = 2.0 if start == MONTHS[-1] else 10.0 if start == MONTHS[-2] else 21.0
        add_metric(
            model_type, "rank_ic_days", days, "month", period, horizon=horizon, run_key=run_key
        )


def test_monthly_ic_returns_the_last_24_months_against_ridge(client: Client, runs: Runs) -> None:
    add_months("lightgbm")
    add_months("ridge")
    add_metric("lightgbm", "rank_ic_nw_t", 9.0, "month", (MONTHS[2], month_end(MONTHS[2])))
    add_metric(
        "lightgbm", "rank_ic", -0.5, "month", (MONTHS[-1], month_end(MONTHS[-1])), segment="liq_t1"
    )
    BacktestMetric.objects.filter(
        model_type="lightgbm", metric="rank_ic", segment="", period_start=MONTHS[-1]
    ).update(value=-0.01)

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
    assert (data[-2]["days"], data[-2]["complete"]) == (10, True)
    assert (data[-1]["days"], data[-1]["complete"], data[-1]["model"]) == (2, False, -0.01)
    assert body["meta"] == {
        "horizon": 20,
        "run_key": "rank-serving",
        "model_type": "lightgbm",
        "baseline": "ridge",
        "source": "self",
        "min_days": 10,
        "above_baseline": 23,
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


def test_before_any_model_serves_the_rank_holdout_is_shown(client: Client, db: None) -> None:
    add_months("lightgbm", run_key="rank-frozen")
    add_months("ridge", run_key="rank-frozen")
    BacktestMetric.objects.filter(model_type="ridge").update(created_at=taipei(2030, 1, 1))
    add_months("lightgbm", run_key="rank-experiment")
    BacktestMetric.objects.filter(run_key="rank-experiment").update(created_at=taipei(2031, 1, 1))
    HoldoutRun.objects.create(scope="rank", attempt=1, run_key="rank-frozen", last_date=OCT_30)

    body = envelope_of(client.get("/api/models/ic-monthly", {"months": "2"}))

    assert [(row["model"], row["baseline"]) for row in body["data"]] == [(0.03, 0.02), (0.03, 0.02)]
    assert (body["meta"]["horizon"], body["meta"]["run_key"]) == (20, "rank-frozen")
    assert (body["meta"]["model_type"], body["meta"]["source"]) == ("lightgbm", "holdout")


def test_a_holdout_with_two_main_model_types_is_not_guessed(client: Client, db: None) -> None:
    add_months("lightgbm", run_key="rank-frozen")
    add_months("xgboost", run_key="rank-frozen")
    HoldoutRun.objects.create(scope="rank", attempt=1, run_key="rank-frozen", last_date=OCT_30)

    body = envelope_of(client.get("/api/models/ic-monthly"))

    assert body["data"] == []
    assert (body["meta"]["run_key"], body["meta"]["source"]) == (None, None)


def test_unfound_research_is_an_empty_chart_not_a_guess(client: Client, runs: Runs) -> None:
    add_months("lightgbm", run_key="rank-experiment")

    body = envelope_of(client.get("/api/models/ic-monthly"))

    assert body["data"] == []
    assert (body["meta"]["run_key"], body["meta"]["source"]) == (None, None)


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

    with django_assert_max_num_queries(6):
        client.get("/api/models/ic-monthly")


SHADOW_GATE = {
    "name": "shadow_oos",
    "reason": "未達門檻",
    "value": -0.013,
    "threshold": "rank IC > 0",
    "sample": 105,
    "sample_unit": "days",
}


def held(objective: str, horizon: int, *gates: dict[str, Any]) -> dict[str, Any]:
    return {
        "slot": f"{objective} {horizon} 日",
        "objective": objective,
        "horizon": horizon,
        "run_id": f"{objective}-2026-11.h{horizon}",
        "gates": list(gates),
    }


def promotion_row(
    scope: str, run_key: object, held_slots: object, day: date = OCT_30
) -> DataQualityLog:
    detail = {
        "scope": scope,
        "run_key": run_key,
        "promoted": True,
        "forced": False,
        "reason": None,
        "failed": [],
        "promoted_slots": [],
        "held_slots": held_slots,
    }
    return DataQualityLog.objects.create(
        date=day, check_name="model_promotion", severity="warning", detail=detail
    )


def not_live(client: Client) -> list[dict[str, Any]]:
    listed: list[dict[str, Any]] = envelope_of(client.get("/api/models"))["meta"]["not_live"]
    return listed


def test_a_horizon_held_back_with_nothing_serving_is_listed_with_its_gates(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")
    promotion_row("rank", "rank-2026-11", [held("rank", 20, {**SHADOW_GATE, "note": "internal"})])

    body = envelope_of(client.get("/api/models"))

    assert body["meta"]["not_live"] == [
        {
            "objective": "rank",
            "horizon": 20,
            "run_key": "rank-2026-11",
            "run_id": "rank-2026-11.h20",
            "date": "2026-10-30",
            "gates": [SHADOW_GATE],
        }
    ]
    assert [(row["objective"], row["horizon"]) for row in body["data"]][:2] == [
        ("rank", 5),
        ("regression", 5),
    ]


def test_a_held_horizon_that_still_has_a_serving_run_is_not_listed(
    client: Client, runs: Runs
) -> None:
    promotion_row("rank", "rank-2026-12", [held("rank", 5, SHADOW_GATE)])

    assert not_live(client) == []


def test_nothing_is_listed_before_any_promotion_is_recorded(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")

    assert not_live(client) == []


def test_each_scope_s_latest_promotion_is_the_one_read(client: Client, runs: Runs) -> None:
    waiting = [runs.rank20.pk, runs.quantile.pk, runs.vol.pk, runs.prob.pk]
    ModelRun.objects.filter(pk__in=waiting).update(status="candidate")
    promotion_row(
        "rank", "rank-2026-10", [held("rank", 20, {**SHADOW_GATE, "value": -0.5})], OCT_29
    )
    promotion_row(
        "interval",
        "interval-2026-11",
        [
            held("classification", 20),
            held("quantile", 20, {**SHADOW_GATE, "name": "data_quality"}),
            held("volatility", 20),
        ],
    )
    promotion_row("rank", "rank-2026-11", [held("rank", 20, SHADOW_GATE)])

    listed = not_live(client)

    assert [(entry["objective"], entry["horizon"], entry["run_key"]) for entry in listed] == [
        ("rank", 20, "rank-2026-11"),
        ("quantile", 20, "interval-2026-11"),
        ("volatility", 20, "interval-2026-11"),
        ("classification", 20, "interval-2026-11"),
    ]
    assert listed[0]["gates"] == [SHADOW_GATE]
    assert listed[1]["gates"][0]["name"] == "data_quality"
    assert listed[2]["gates"] == []


def test_a_later_promotion_that_held_nothing_clears_the_older_one(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")
    promotion_row("rank", "rank-2026-10", [held("rank", 20, SHADOW_GATE)], OCT_29)
    promotion_row("rank", "rank-2026-11", [])

    assert not_live(client) == []


def test_a_promotion_row_from_before_the_lists_names_nothing(
    client: Client, runs: Runs, caplog: pytest.LogCaptureFixture
) -> None:
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")
    DataQualityLog.objects.create(
        date=OCT_30,
        check_name="model_promotion",
        severity="warning",
        detail={"scope": "rank", "run_key": "rank-2026-11", "promoted": False, "failed": []},
    )

    assert not_live(client) == []
    assert "unreadable" not in caplog.text


@pytest.mark.parametrize(
    "held_slots",
    [
        None,
        "rank 20 日",
        [None],
        [{"objective": "rank"}],
        [{**held("rank", 20, SHADOW_GATE), "horizon": 60}],
        [{**held("rank", 20, SHADOW_GATE), "horizon": "20"}],
        [{**held("rank", 20, SHADOW_GATE), "objective": "screening"}],
        [{**held("rank", 20), "gates": "shadow_oos"}],
        [{**held("rank", 20), "gates": [{"reason": "沒有名字"}]}],
    ],
)
def test_held_entries_that_cannot_be_read_are_left_out(
    client: Client, runs: Runs, held_slots: object, caplog: pytest.LogCaptureFixture
) -> None:
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")
    row = promotion_row("rank", "rank-2026-11", held_slots)

    assert not_live(client) == []
    assert f"promotion row {row.id} has unreadable held_slots" in caplog.text


def test_keys_that_are_not_text_come_back_empty(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")
    promotion_row("rank", 202611, [{**held("rank", 20, SHADOW_GATE), "run_id": ["x"]}])

    listed = not_live(client)

    assert [(entry["run_key"], entry["run_id"]) for entry in listed] == [(None, None)]


def test_without_a_horizon_the_chart_follows_the_ranking_that_serves(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")
    add_months("lightgbm")
    add_months("lightgbm", horizon=5)
    BacktestMetric.objects.filter(horizon=5, metric="rank_ic").update(value=0.01)

    body = envelope_of(client.get("/api/models/ic-monthly", {"months": "3"}))

    assert (body["meta"]["horizon"], body["meta"]["source"]) == (5, "self")
    assert [row["model"] for row in body["data"]] == [0.01, 0.01, 0.01]


def test_an_asked_horizon_is_charted_even_when_its_ranking_does_not_serve(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")
    add_months("lightgbm", run_key="rank-frozen")
    add_months("lightgbm", horizon=5)
    HoldoutRun.objects.create(scope="rank", attempt=1, run_key="rank-frozen", last_date=OCT_30)

    body = envelope_of(client.get("/api/models/ic-monthly", {"horizon": "20", "months": "2"}))

    assert (body["meta"]["horizon"], body["meta"]["source"]) == (20, "holdout")
    assert body["meta"]["run_key"] == "rank-frozen"


def test_with_both_rankings_serving_the_chart_stays_on_twenty_days(
    client: Client, runs: Runs
) -> None:
    add_months("lightgbm")
    add_months("lightgbm", horizon=5)

    assert envelope_of(client.get("/api/models/ic-monthly"))["meta"]["horizon"] == 20


def test_the_api_reads_the_check_name_the_research_layer_writes() -> None:
    assert serving.PROMOTION_CHECK == promotion.PROMOTION_CHECK


def test_one_unreadable_held_entry_does_not_drop_the_readable_ones(
    client: Client, runs: Runs, caplog: pytest.LogCaptureFixture
) -> None:
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")
    row = promotion_row("rank", "rank-2026-11", [held("rank", 20, SHADOW_GATE), None])

    listed = not_live(client)

    assert [(entry["objective"], entry["horizon"]) for entry in listed] == [("rank", 20)]
    assert f"promotion row {row.id} has unreadable held_slots" in caplog.text


def test_the_row_written_last_is_read_even_when_it_is_dated_earlier(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")
    promotion_row("rank", "rank-first", [held("rank", 20, {**SHADOW_GATE, "value": -0.5})], OCT_30)
    promotion_row("rank", "rank-second", [held("rank", 20, SHADOW_GATE)], OCT_29)

    listed = not_live(client)

    assert [(entry["run_key"], entry["date"]) for entry in listed] == [
        ("rank-second", "2026-10-29")
    ]
    assert listed[0]["gates"] == [SHADOW_GATE]


def test_an_unreadable_row_is_warned_about_once_not_on_every_request(
    client: Client, runs: Runs, caplog: pytest.LogCaptureFixture
) -> None:
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")
    promotion_row("rank", "rank-2026-11", [None])

    assert (not_live(client), not_live(client)) == ([], [])
    assert caplog.text.count("has unreadable held_slots") == 1


def test_the_api_urls_load_without_the_training_modules() -> None:
    probe = (
        "import django, sys; django.setup(); import apps.api.urls; "
        "heavy = ('lightgbm', 'apps.research.promotion'); "
        "print(sorted(name for name in heavy if name in sys.modules))"
    )

    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert done.stdout.strip() == "[]"
