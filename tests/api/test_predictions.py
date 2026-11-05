import math
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.market.models import Market
from apps.research.models import ModelMonitor, ModelRun
from tests.api.conftest import (
    add_bars,
    add_model_run,
    add_predictions,
    add_stock,
    envelope_of,
    sessions_until,
    taipei,
)

pytestmark = pytest.mark.usefixtures("calendar_in_db")

OCT_28, OCT_29, OCT_30 = date(2026, 10, 28), date(2026, 10, 29), date(2026, 10, 30)


@dataclass(frozen=True)
class Runs:
    rank20: ModelRun
    quantile: ModelRun
    vol: ModelRun
    prob: ModelRun
    rank5: ModelRun
    reg5: ModelRun


def seed_market() -> None:
    add_stock()
    add_stock(stock_id="2317", name="鴻海", industry="其他電子業")
    add_stock(stock_id="6488", name="環球晶", market=Market.TPEX)
    add_stock(stock_id="8299", name="群聯", market=Market.TPEX)
    add_bars("2330", [OCT_29, OCT_30], [990.0, 1000.0])
    add_bars("2317", [OCT_29, OCT_30], [198.0, 200.0])
    add_bars("6488", [OCT_29, OCT_30], [490.0, 500.0])
    add_bars("8299", [OCT_29, OCT_30], [800.0, None])


@pytest.fixture
def runs(db: None) -> Runs:
    seed_market()
    seeded = Runs(
        rank20=add_model_run("rank-h20", "rank"),
        quantile=add_model_run("quantile-h20", "quantile"),
        vol=add_model_run("vol-h20", "volatility"),
        prob=add_model_run("prob-h20", "classification"),
        rank5=add_model_run("rank-h5", "rank"),
        reg5=add_model_run("reg-h5", "regression"),
    )
    add_predictions(
        seeded.rank20,
        OCT_30,
        20,
        {
            "2330": {"score": 0.9, "rank": 1, "pct": 0.99},
            "6488": {"score": 0.5, "rank": 2, "pct": 0.5},
            "2317": {"score": 0.1, "rank": 3, "pct": 0.01},
            "8299": {"score": 0.3, "in_universe": False},
        },
    )
    add_predictions(
        seeded.rank20,
        OCT_29,
        20,
        {
            "6488": {"score": 0.8, "rank": 1, "pct": 0.99},
            "2330": {"score": 0.4, "rank": 2, "pct": 0.5},
            "2317": {"score": 0.1, "rank": 3, "pct": 0.01},
        },
    )
    add_predictions(
        seeded.quantile,
        OCT_30,
        20,
        {
            "2330": {"q10": -0.05, "q50": 0.04, "q90": 0.12, "ret_pred": 0.04},
            "2317": {"q10": -0.1, "q50": -0.02, "q90": 0.05, "ret_pred": -0.02},
            "8299": {"q10": -0.2, "q50": 0.0, "q90": 0.2, "ret_pred": 0.0, "in_universe": False},
        },
    )
    add_predictions(
        seeded.vol,
        OCT_30,
        20,
        {code: {"vol_pred": 0.3} for code in ("2330", "2317", "6488", "8299")},
    )
    add_predictions(
        seeded.prob,
        OCT_30,
        20,
        {
            "2330": {"prob_up_5pct": 0.6},
            "2317": {"prob_up_5pct": 0.2},
            "6488": {"prob_up_5pct": 0.4},
        },
    )
    add_predictions(
        seeded.rank5,
        OCT_30,
        5,
        {
            "2317": {"score": 0.7, "rank": 1, "pct": 0.99},
            "2330": {"score": 0.2, "rank": 2, "pct": 0.5},
            "6488": {"score": 0.1, "rank": 3, "pct": 0.01},
        },
    )
    add_predictions(
        seeded.reg5,
        OCT_30,
        5,
        {"2317": {"ret_pred": 0.02}, "2330": {"ret_pred": 0.01}, "6488": {"ret_pred": -0.01}},
    )
    return seeded


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
            "source": "quantile",
        },
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
    }
    assert meta["degraded"] == []


def test_a_stock_without_quantiles_gets_the_volatility_band(client: Client, runs: Runs) -> None:
    row = by_code(envelope_of(client.get("/api/predictions")))["6488"]
    spread = 1.28 * 0.3 * math.sqrt(20 / 252)

    assert (row["ret_pred"], row["price_pred"]) == (None, None)
    assert row["interval"] == {
        "low": round(500 * math.exp(-spread), 4),
        "high": round(500 * math.exp(spread), 4),
        "ret_low": round(math.exp(-spread) - 1, 6),
        "ret_high": round(math.exp(spread) - 1, 6),
        "source": "volatility",
    }


def test_untraded_stock_is_priced_on_its_last_close_and_ranked_last(
    client: Client, runs: Runs
) -> None:
    row = by_code(envelope_of(client.get("/api/predictions")))["8299"]

    assert (row["close"], row["close_date"]) == (800.0, "2026-10-29")
    assert (row["rank"], row["in_universe"], row["prev_rank"]) == (None, False, None)
    assert row["interval"]["source"] == "quantile"
    assert row["prob_up_5pct"] is None


def test_five_day_rows_use_the_regression_and_a_five_day_band(client: Client, runs: Runs) -> None:
    body = envelope_of(client.get("/api/predictions", {"horizon": "5"}))
    spread = 1.28 * 0.3 * math.sqrt(5 / 252)

    assert [row["stock_id"] for row in body["data"]] == ["2317", "2330", "6488"]
    first = body["data"][0]
    assert (first["ret_pred"], first["price_pred"], first["prob_up_5pct"]) == (0.02, 204.0, None)
    assert first["interval"]["source"] == "volatility"
    assert first["interval"]["high"] == round(200 * math.exp(spread), 4)
    assert [model["role"] for model in body["meta"]["models"]] == ["rank", "return", "volatility"]


def test_a_degraded_quantile_model_switches_to_the_volatility_band(
    client: Client, runs: Runs
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")

    body = envelope_of(client.get("/api/predictions"))
    row = by_code(body)["2330"]

    assert row["ret_pred"] == 0.04
    assert row["interval"]["source"] == "volatility"
    assert body["meta"]["degraded"] == ["return", "interval"]


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
    candidate = add_model_run("rank-candidate", "rank", status="candidate")
    add_predictions(candidate, date(2026, 11, 2), 20, {"2330": {"score": 0.1, "rank": 1, "pct": 1}})

    earlier = envelope_of(client.get("/api/predictions", {"date": "2026-10-29"}))
    latest = envelope_of(client.get("/api/predictions"))

    assert latest["meta"]["date"] == "2026-10-30"
    assert earlier["meta"]["date"] == "2026-10-29"
    assert [row["stock_id"] for row in earlier["data"]] == ["6488", "2330", "2317"]
    assert earlier["data"][0]["prev_rank"] is None


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
    days = sessions_until(OCT_30)[-61:]
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.rank20, date=day, horizon=20, ic=0.5 if n == 0 else 0.04, n_stocks=3)
        for n, day in enumerate(days)
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=runs.quantile, date=day, horizon=20, coverage=0.79, n_stocks=3)
        for day in days[-10:]
    )

    monitor = envelope_of(client.get("/api/predictions"))["meta"]["monitor"]

    assert monitor == {"rank_ic": 0.04, "rank_ic_days": 60, "coverage": 0.79, "coverage_days": 10}


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


def test_predictions_query_count(
    client: Client, runs: Runs, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    with django_assert_max_num_queries(12):
        client.get("/api/predictions")


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
