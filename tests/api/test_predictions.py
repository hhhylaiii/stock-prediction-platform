import math
import time
from datetime import date
from typing import Any

import pytest
from django.db import connection
from django.test import Client
from django.test.utils import CaptureQueriesContext
from pytest_django import DjangoAssertNumQueries

from apps.api.serving import HOLDOUT_RELEASE
from apps.market.models import TradingCalendar
from apps.research.holdout import DEGRADED, FAILED, PASSED, PRODUCTION
from apps.research.models import FeatureDaily, ModelMonitor, ModelRun, Prediction
from tests.api.conftest import (
    CALENDAR_START,
    EARLY_CALENDAR_START,
    NOV_2,
    OCT_28,
    OCT_29,
    OCT_30,
    RELEASE_REASON,
    RELEASE_REPORT,
    Runs,
    add_bars,
    add_model_run,
    add_predictions,
    add_stock,
    envelope_of,
    release_params,
    sessions_until,
    taipei,
    trained_until,
)

pytestmark = pytest.mark.usefixtures("calendar_in_db")


@pytest.fixture
def calendar_start() -> date:
    """Runs here train before July, so their out-of-sample days start inside the calendar."""
    return EARLY_CALENDAR_START


def by_code(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["stock_id"]: row for row in body["data"]}


def test_twenty_day_rows_merge_every_model_in_rank_order(client: Client, runs: Runs) -> None:
    response = client.get("/api/predictions")

    assert response.status_code == 200
    body = envelope_of(response)
    assert [row["stock_id"] for row in body["data"]] == ["2330", "6488", "2317", "8299"]
    assert body["data"][0] == {
        "stock_id": "2330",
        "name": "台積電",
        "market": "TWSE",
        "market_label": "上市",
        "rank": 1,
        "pct": 0.99,
        "score": 0.9,
        "prev_rank": 2,
        "in_universe": True,
        "close": 1000.0,
        "close_date": "2026-10-30",
        "ret_pred": 0.04,
        "price_pred": 1040.0,
        "interval": {
            "low": 950.0,
            "high": 1120.0,
            "ret_low": -0.05,
            "ret_high": 0.12,
        },
        "interval_source": "quantile",
        "prob_up_5pct": 0.6,
        "vol_pred": 0.3,
    }


def test_meta_names_the_date_universe_and_serving_models(client: Client, runs: Runs) -> None:
    meta = envelope_of(client.get("/api/predictions"))["meta"]

    assert meta["date"] == "2026-10-30"
    assert meta["horizon"] == 20
    assert meta["universe"] == 3
    assert (meta["total"], meta["page"], meta["page_size"], meta["pages"]) == (4, 1, 20, 1)
    assert [(model["role"], model["run_id"]) for model in meta["models"]] == [
        ("rank", "rank-h20"),
        ("return", "quantile-h20"),
        ("interval", "quantile-h20"),
        ("volatility", "vol-h20"),
        ("probability", "prob-h20"),
    ]
    assert meta["models"][0] == {
        "role": "rank",
        "run_id": "rank-h20",
        "objective": "rank",
        "horizon": 20,
        "model_type": "lightgbm",
        "version": "2026-10",
        "status": "active",
        "train_start": "2023-10-01",
        "train_end": "2026-09-30",
        "released_verdict": None,
    }
    assert (meta["degraded"], meta["interval_alert"]) == ([], False)


def verdicts_of(body: dict[str, Any]) -> dict[str, str | None]:
    return {model["role"]: model["released_verdict"] for model in body["meta"]["models"]}


@pytest.mark.parametrize("status", [FAILED, DEGRADED])
def test_a_model_built_on_a_released_holdout_names_the_verdict_and_not_the_reason(
    client: Client, runs: Runs, status: str
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(params=release_params(status))

    response = client.get("/api/predictions")

    assert verdicts_of(envelope_of(response)) == {
        "rank": None,
        "return": status,
        "interval": status,
        "volatility": None,
        "probability": None,
    }
    body = response.content.decode()
    assert RELEASE_REASON not in body
    assert RELEASE_REPORT not in body


def test_the_released_verdict_is_read_once_per_request_not_per_prediction_row(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(params=release_params(FAILED))

    with CaptureQueriesContext(connection) as queries:
        body = envelope_of(client.get("/api/predictions"))

    reading = [query["sql"] for query in queries if HOLDOUT_RELEASE in query["sql"]]
    assert verdicts_of(body)["rank"] == FAILED
    assert len(reading) == 1
    assert '"prediction"' not in reading[0]


def test_a_release_the_model_page_cannot_show_is_not_served_as_released(
    client: Client, runs: Runs, caplog: pytest.LogCaptureFixture
) -> None:
    half = {PRODUCTION: {HOLDOUT_RELEASE: {"status": FAILED, "reason": "half a record"}}}
    ModelRun.objects.filter(pk="rank-h20").update(params=half)

    verdicts = verdicts_of(envelope_of(client.get("/api/predictions")))
    listed = envelope_of(client.get("/api/models"))["data"]

    assert verdicts["rank"] is None
    assert [row["holdout_release"] for row in listed if row["run_id"] == "rank-h20"] == [None]
    assert "rank-h20" in caplog.text
    assert "holdout release" in caplog.text


@pytest.mark.parametrize("status", [PASSED, "released", 1, None, [FAILED]])
def test_a_status_no_release_can_name_is_not_served_as_released(
    client: Client, runs: Runs, status: object
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(params=release_params(status))

    assert verdicts_of(envelope_of(client.get("/api/predictions")))["rank"] is None


def test_production_params_without_a_release_are_not_served_as_released(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(params={PRODUCTION: "rank-2026-11"})
    ModelRun.objects.filter(pk="vol-h20").update(params={PRODUCTION: {HOLDOUT_RELEASE: FAILED}})

    verdicts = verdicts_of(envelope_of(client.get("/api/predictions")))

    assert (verdicts["rank"], verdicts["volatility"]) == (None, None)


def test_a_stock_without_quantiles_gets_the_volatility_band(client: Client, runs: Runs) -> None:
    add_predictions(
        runs.quantile, OCT_30, 5, {"6488": {"q10": -0.5, "q50": 0.1, "q90": 0.5, "ret_pred": 0.1}}
    )

    row = by_code(envelope_of(client.get("/api/predictions")))["6488"]
    spread = 1.28 * 0.3 * math.sqrt(20 / 252)

    assert (row["ret_pred"], row["price_pred"]) == (None, None)
    assert row["interval"] == {
        "low": round(500 * (1 - spread), 4),
        "high": round(500 * (1 + spread), 4),
        "ret_low": round(-spread, 6),
        "ret_high": round(spread, 6),
    }
    assert row["interval_source"] == "volatility"


def test_untraded_stock_is_priced_on_its_last_close_and_ranked_last(
    client: Client, runs: Runs
) -> None:
    row = by_code(envelope_of(client.get("/api/predictions")))["8299"]

    assert (row["close"], row["close_date"]) == (800.0, "2026-10-29")
    assert (row["rank"], row["in_universe"], row["prev_rank"]) == (None, False, None)
    assert row["interval_source"] == "quantile"
    assert row["prob_up_5pct"] is None


def test_five_day_rows_use_the_regression_and_a_five_day_band(client: Client, runs: Runs) -> None:
    body = envelope_of(client.get("/api/predictions", {"horizon": "5"}))
    spread = 1.28 * 0.3 * math.sqrt(5 / 252)

    assert [row["stock_id"] for row in body["data"]] == ["2317", "2330", "6488"]
    first = body["data"][0]
    assert (first["ret_pred"], first["price_pred"], first["prob_up_5pct"]) == (0.02, 204.0, None)
    assert first["interval_source"] == "volatility"
    assert first["interval"]["high"] == round(200 * (1.02 + spread), 4)
    assert first["interval"]["low"] < first["price_pred"] < first["interval"]["high"]
    assert [model["role"] for model in body["meta"]["models"]] == ["rank", "return", "volatility"]


def test_a_degraded_quantile_model_switches_to_the_volatility_band(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")

    body = envelope_of(client.get("/api/predictions"))
    row = by_code(body)["2330"]
    spread = 1.28 * 0.3 * math.sqrt(20 / 252)

    assert (row["ret_pred"], row["price_pred"]) == (0.04, 1040.0)
    assert row["interval_source"] == "volatility"
    assert (row["interval"]["low"], row["interval"]["high"]) == (
        round(1000 * (1.04 - spread), 4),
        round(1000 * (1.04 + spread), 4),
    )
    for each in body["data"]:
        if each["price_pred"] is not None and each["interval"] is not None:
            assert each["interval"]["low"] <= each["price_pred"] <= each["interval"]["high"]
    assert body["meta"]["degraded"] == ["return", "interval"]


def test_without_a_volatility_prediction_the_band_uses_tech_vol_20d(
    client: Client, runs: Runs
) -> None:
    Prediction.objects.filter(run=runs.vol, stock_id__in=["6488", "2317"]).delete()
    FeatureDaily.objects.create(stock_id="6488", date=OCT_30, in_universe=True, tech_vol_20d=0.5)
    FeatureDaily.objects.create(stock_id="6488", date=OCT_29, in_universe=True, tech_vol_20d=0.9)
    spread = 1.28 * 0.5 * math.sqrt(20 / 252)

    rows = by_code(envelope_of(client.get("/api/predictions")))

    assert rows["6488"]["interval_source"] == "tech_vol_20d"
    assert rows["6488"]["vol_pred"] is None
    assert rows["6488"]["interval"]["high"] == round(500 * (1 + spread), 4)
    assert rows["2317"]["interval_source"] == "quantile"


def test_an_alerted_quantile_model_without_vol_pred_uses_tech_vol_20d(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")
    Prediction.objects.filter(run=runs.vol, stock_id="2330").delete()
    FeatureDaily.objects.create(stock_id="2330", date=OCT_30, in_universe=True, tech_vol_20d=0.5)

    row = by_code(envelope_of(client.get("/api/predictions")))["2330"]

    assert row["interval_source"] == "tech_vol_20d"
    assert row["interval"]["low"] <= row["price_pred"] <= row["interval"]["high"]


def test_no_volatility_at_all_leaves_the_band_empty(client: Client, runs: Runs) -> None:
    Prediction.objects.filter(run=runs.vol, stock_id="6488").delete()

    row = by_code(envelope_of(client.get("/api/predictions")))["6488"]

    assert (row["interval"], row["interval_source"]) == (None, None)


def test_candidate_retired_and_screening_rows_do_not_serve(client: Client, runs: Runs) -> None:
    flipped = {"2317": {"score": 1.0, "rank": 1, "pct": 0.99}}
    add_predictions(
        add_model_run("rank-candidate", "rank", status="candidate"), OCT_30, 20, flipped
    )
    add_predictions(add_model_run("rank-old", "rank", status="retired"), OCT_30, 20, flipped)
    add_predictions(
        add_model_run("screening", "rank", model_type="screening", created_at=taipei(2026, 11, 1)),
        OCT_30,
        20,
        flipped,
    )

    body = envelope_of(client.get("/api/predictions"))

    assert body["data"][0]["stock_id"] == "2330"
    assert body["meta"]["models"][0]["run_id"] == "rank-h20"


def test_an_active_run_outranks_a_newer_degraded_one(client: Client, runs: Runs) -> None:
    newer = add_model_run("rank-new", "rank", status="degraded", created_at=taipei(2026, 10, 20))
    add_predictions(newer, OCT_30, 20, {"2317": {"score": 1.0, "rank": 1, "pct": 0.99}})

    body = envelope_of(client.get("/api/predictions"))

    assert body["meta"]["models"][0]["run_id"] == "rank-h20"
    assert body["data"][0]["stock_id"] == "2330"


def test_the_newer_of_two_active_runs_serves(client: Client, runs: Runs) -> None:
    newer = add_model_run("rank-new", "rank", created_at=taipei(2026, 10, 20))
    add_predictions(newer, OCT_30, 20, {"2317": {"score": 1.0, "rank": 1, "pct": 0.99}})

    body = envelope_of(client.get("/api/predictions"))

    assert body["meta"]["models"][0]["run_id"] == "rank-new"
    assert [row["stock_id"] for row in body["data"]] == ["2317"]


def test_retired_rows_still_answer_an_earlier_date(client: Client, runs: Runs) -> None:
    old = add_model_run("rank-old", "rank", status="retired")
    add_predictions(old, OCT_28, 20, {"2317": {"score": 0.9, "rank": 1, "pct": 0.99}})

    body = envelope_of(client.get("/api/predictions", {"date": "2026-10-28"}))

    assert [row["stock_id"] for row in body["data"]] == ["2317"]
    assert body["meta"]["models"][0]["status"] == "retired"


def test_date_defaults_to_the_latest_served_date(client: Client, runs: Runs) -> None:
    later = {"2330": {"score": 0.1, "rank": 1, "pct": 1}}
    add_predictions(add_model_run("rank-candidate", "rank", status="candidate"), NOV_2, 20, later)
    add_predictions(add_model_run("rank-old", "rank", status="retired"), NOV_2, 20, later)
    add_predictions(add_model_run("quantile-new", "quantile"), NOV_2, 20, later)

    earlier = envelope_of(client.get("/api/predictions", {"date": "2026-10-29"}))
    latest = envelope_of(client.get("/api/predictions"))

    assert latest["meta"]["date"] == "2026-10-30"
    assert earlier["meta"]["date"] == "2026-10-29"
    assert [row["stock_id"] for row in earlier["data"]] == ["6488", "2330", "2317"]
    assert earlier["data"][0]["prev_rank"] is None
    assert (by_code(earlier)["2330"]["close"], by_code(earlier)["2330"]["close_date"]) == (
        990.0,
        "2026-10-29",
    )


def test_previous_rank_comes_from_the_run_that_served_that_day(client: Client, runs: Runs) -> None:
    old = add_model_run("rank-old", "rank", status="retired")
    add_predictions(old, OCT_29, 20, {"8299": {"score": 0.0, "rank": 9, "pct": 0.0}})

    rows = by_code(envelope_of(client.get("/api/predictions")))

    assert (rows["2330"]["prev_rank"], rows["8299"]["prev_rank"]) == (2, None)


def test_on_a_model_switch_the_previous_rank_comes_from_the_retired_run(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(status="retired")
    switched = add_model_run("rank-new", "rank", created_at=taipei(2026, 10, 30))
    add_predictions(switched, OCT_30, 20, {"2330": {"score": 1.0, "rank": 1, "pct": 1.0}})

    body = envelope_of(client.get("/api/predictions"))

    assert body["meta"]["models"][0]["run_id"] == "rank-new"
    assert by_code(body)["2330"]["prev_rank"] == 2


def test_a_broken_calendar_drops_the_previous_rank_and_logs_the_empty_monitor(
    client: Client, runs: Runs, caplog: pytest.LogCaptureFixture
) -> None:
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.rank20, date=day, horizon=20, ic=0.05, n_stocks=3)
        for day in (OCT_29, OCT_30)
    )
    TradingCalendar.objects.filter(date=date(2026, 8, 3)).delete()

    response = client.get("/api/predictions")

    assert response.status_code == 200
    body = envelope_of(response)
    rows = by_code(body)
    assert (rows["2330"]["rank"], rows["2330"]["prev_rank"]) == (1, None)
    assert body["meta"]["monitor"]["rank_ic"] is None
    assert [r.getMessage() for r in caplog.records if r.levelname == "ERROR"] == [
        "trading calendar cannot be read (trading_calendar has 1 missing day(s));"
        " no monitor row counts as out of sample"
    ]


@pytest.mark.parametrize(
    ("params", "codes"),
    [
        ({"market": "TPEx"}, ["6488", "8299"]),
        ({"q": "23"}, ["2330", "2317"]),
        ({"q": "鴻"}, ["2317"]),
        ({"q": "  環球 "}, ["6488"]),
        ({"q": "nothing"}, []),
    ],
)
def test_filters_by_market_and_search(
    client: Client, runs: Runs, params: dict[str, str], codes: list[str]
) -> None:
    body = envelope_of(client.get("/api/predictions", params))

    assert [row["stock_id"] for row in body["data"]] == codes
    assert body["meta"]["total"] == len(codes)
    assert body["meta"]["universe"] == 3


def test_no_predictions_yet_is_an_empty_page(client: Client, db: None) -> None:
    body = envelope_of(client.get("/api/predictions"))

    assert body["data"] == []
    assert body["meta"]["date"] is None
    assert body["meta"]["models"] == []
    assert body["meta"]["total"] == 0


def test_monitor_averages_the_last_60_evaluated_dates(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk__in=["rank-h20", "quantile-h20"]).update(train_end=date(2026, 6, 30))
    days = sessions_until(OCT_30)[-62:]
    ics = {0: 0.5, len(days) - 1: None}
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.rank20, date=day, horizon=20, ic=ics.get(n, 0.04), n_stocks=3)
        for n, day in enumerate(days)
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.rank20, date=day, horizon=20, ic=0.9, n_stocks=3)
        for day in (NOV_2, date(2026, 11, 3))
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.quantile, date=day, horizon=20, coverage=0.79, n_stocks=3)
        for day in days[-10:]
    )

    monitor = envelope_of(client.get("/api/predictions"))["meta"]["monitor"]

    assert monitor == {"rank_ic": 0.04, "rank_ic_days": 60, "coverage": 0.79, "coverage_days": 10}


def test_monitor_chains_the_runs_that_served_and_skips_in_sample_days(
    client: Client, runs: Runs
) -> None:
    switch = date(2026, 9, 30)
    days = sessions_until(OCT_30)[-30:]
    ModelRun.objects.filter(pk="rank-h20").update(train_end=trained_until(switch))
    old = add_model_run("rank-old", "rank", status="retired", train_end=date(2026, 6, 30))
    ModelMonitor.objects.bulk_create(
        ModelMonitor(
            run=old if day <= switch else runs.rank20,
            date=day,
            horizon=20,
            ic=0.02 if day <= switch else 0.06,
            n_stocks=3,
        )
        for day in days
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.rank20, date=day, horizon=20, ic=0.9, n_stocks=3)
        for day in days
        if day <= switch
    )

    monitor = envelope_of(client.get("/api/predictions"))["meta"]["monitor"]

    after = sum(day > switch for day in days)
    expected = (0.02 * (len(days) - after) + 0.06 * after) / len(days)
    assert (monitor["rank_ic"], monitor["rank_ic_days"]) == (round(expected, 4), len(days))


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"horizon": "10"}, "horizon"),
        ({"market": "OTC"}, "market"),
        ({"date": "2026/10/30"}, "date"),
        ({"q": "x" * 33}, "q"),
    ],
)
def test_bad_queries_are_invalid(client: Client, params: dict[str, str], field: str) -> None:
    response = client.get("/api/predictions", params)

    assert response.status_code == 400
    assert set(envelope_of(response)["error"]["fields"]) == {field}


def test_predictions_query_count_on_the_tech_vol_path(
    client: Client, runs: Runs, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    Prediction.objects.filter(run=runs.vol, stock_id="6488").delete()
    FeatureDaily.objects.create(stock_id="6488", date=OCT_30, in_universe=True, tech_vol_20d=0.5)
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=run, date=day, horizon=20, ic=0.05, coverage=0.8, n_stocks=3)
        for run in (runs.rank20, runs.quantile)
        for day in (OCT_29, OCT_30)
    )

    with django_assert_max_num_queries(14):
        response = client.get("/api/predictions")

    body = envelope_of(response)
    assert by_code(body)["6488"]["interval_source"] == "tech_vol_20d"
    assert (body["meta"]["monitor"]["rank_ic_days"], body["meta"]["monitor"]["coverage_days"]) == (
        2,
        2,
    )


def test_two_thousand_stocks_answer_within_half_a_second(client: Client, db: None) -> None:
    codes = [f"{9000 + n}" for n in range(2000)]
    for code in codes:
        add_stock(stock_id=code, name=f"股{code}")
    for code in codes:
        add_bars(code, [OCT_29, OCT_30], [100.0, 101.0])
    rank = add_model_run("rank-h20", "rank")
    quantile = add_model_run("quantile-h20", "quantile")
    add_predictions(
        rank,
        OCT_30,
        20,
        {code: {"score": -n, "rank": n + 1, "pct": 1 - n / 2000} for n, code in enumerate(codes)},
    )
    add_predictions(
        rank,
        OCT_29,
        20,
        {code: {"score": n, "rank": 2000 - n, "pct": n / 2000} for n, code in enumerate(codes)},
    )
    add_predictions(
        quantile,
        OCT_30,
        20,
        {code: {"q10": -0.1, "q50": 0.0, "q90": 0.1, "ret_pred": 0.0} for code in codes},
    )
    client.get("/api/predictions")

    started = time.perf_counter()
    response = client.get("/api/predictions", {"q": "股9"})
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert envelope_of(response)["meta"]["total"] == 1000
    assert elapsed < 0.5


@pytest.mark.parametrize("calendar_start", [CALENDAR_START])
def test_the_first_calendar_session_has_no_previous_rank(client: Client, db: None) -> None:
    add_stock()
    add_bars("2330", [date(2026, 7, 1)], [900.0])
    add_predictions(
        add_model_run("rank-h20", "rank"),
        date(2026, 7, 1),
        20,
        {"2330": {"score": 1.0, "rank": 1, "pct": 1.0}},
    )

    row = envelope_of(client.get("/api/predictions"))["data"][0]

    assert (row["rank"], row["prev_rank"], row["close"]) == (1, None, 900.0)


@pytest.mark.parametrize(
    ("degraded_on", "source"), [("2026-10-30", "volatility"), ("2026-11-02", "quantile")]
)
def test_a_degraded_band_counts_from_its_degraded_on_day(
    client: Client, runs: Runs, degraded_on: str, source: str
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(
        status="degraded", params={"degraded_on": degraded_on}
    )

    body = envelope_of(client.get("/api/predictions"))

    assert by_code(body)["8299"]["interval_source"] == source
    assert body["meta"]["interval_alert"] is (source == "volatility")
