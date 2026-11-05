from typing import Any

import pytest
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.research.models import FeatureDaily, ModelRun, Prediction
from tests.api.conftest import (
    NOV_2,
    OCT_29,
    OCT_30,
    Runs,
    add_predictions,
    add_stock,
    envelope_of,
)

pytestmark = pytest.mark.usefixtures("calendar_in_db")


def horizons_of(body: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {entry["horizon"]: entry for entry in body["data"]["horizons"]}


def test_both_horizons_merge_every_serving_model(client: Client, runs: Runs) -> None:
    response = client.get("/api/stocks/2330/predictions")

    assert response.status_code == 200
    body = envelope_of(response)
    assert body["data"]["stock_id"] == "2330"
    five, twenty = horizons_of(body)[5], horizons_of(body)[20]
    assert (five["date"], five["universe"], twenty["date"], twenty["universe"]) == (
        "2026-10-30",
        3,
        "2026-10-30",
        3,
    )
    assert (five["prediction"]["rank"], five["prediction"]["ret_pred"]) == (2, 0.01)
    assert (five["prediction"]["price_pred"], five["prediction"]["prob_up_5pct"]) == (1010.0, None)
    assert five["prediction"]["interval_source"] == "volatility"
    assert twenty["prediction"]["rank"] == 1
    assert twenty["prediction"]["interval"] == {
        "low": 950.0,
        "high": 1120.0,
        "ret_low": -0.05,
        "ret_high": 0.12,
    }
    assert (twenty["prediction"]["prob_up_5pct"], twenty["prediction"]["vol_pred"]) == (0.6, 0.3)
    assert [model["role"] for model in five["models"]] == ["rank", "return", "volatility"]
    assert twenty["models"][1]["run_id"] == "quantile-h20"
    assert (five["degraded"], twenty["degraded"]) == ([], [])


def test_realised_volatility_comes_from_the_twenty_day_prediction_date(
    client: Client, runs: Runs
) -> None:
    add_predictions(runs.rank5, NOV_2, 5, {"2330": {"score": 0.1, "rank": 1, "pct": 1.0}})
    for day, vol in ((OCT_29, 0.1), (OCT_30, 0.25), (NOV_2, 0.9)):
        FeatureDaily.objects.create(
            stock_id="2330", date=day, in_universe=True, tech_vol_20d=vol, tech_vol_60d=vol / 2
        )

    data = envelope_of(client.get("/api/stocks/2330/predictions"))["data"]
    other = envelope_of(client.get("/api/stocks/2317/predictions"))["data"]

    assert horizons_of({"data": data})[5]["date"] == "2026-11-02"
    assert data["realized_vol"] == {
        "date": "2026-10-30",
        "tech_vol_20d": 0.25,
        "tech_vol_60d": 0.125,
    }
    assert other["realized_vol"] is None


def test_without_a_twenty_day_date_realised_volatility_follows_the_five_day_one(
    client: Client, runs: Runs
) -> None:
    Prediction.objects.filter(run=runs.rank20).delete()
    for day, vol in ((OCT_30, 0.25), (NOV_2, 0.9)):
        FeatureDaily.objects.create(stock_id="2330", date=day, in_universe=True, tech_vol_60d=vol)

    data = envelope_of(client.get("/api/stocks/2330/predictions"))["data"]

    assert horizons_of({"data": data})[20]["date"] is None
    assert data["realized_vol"]["date"] == "2026-10-30"


def test_a_degraded_model_is_named_per_horizon(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")

    entries = horizons_of(envelope_of(client.get("/api/stocks/2330/predictions")))

    assert entries[20]["degraded"] == ["return", "interval"]
    assert entries[20]["prediction"]["interval_source"] == "volatility"
    assert entries[5]["degraded"] == []


def test_a_stock_the_models_skipped_has_no_prediction(client: Client, runs: Runs) -> None:
    add_stock(stock_id="1101", name="台泥")

    data = envelope_of(client.get("/api/stocks/1101/predictions"))["data"]

    assert [(entry["date"], entry["prediction"]) for entry in data["horizons"]] == [
        ("2026-10-30", None),
        ("2026-10-30", None),
    ]
    assert data["realized_vol"] is None


def test_an_earlier_date_answers_from_that_day(client: Client, runs: Runs) -> None:
    entries = horizons_of(
        envelope_of(client.get("/api/stocks/2330/predictions", {"date": "2026-10-29"}))
    )

    assert entries[20]["prediction"]["rank"] == 2
    assert (entries[20]["prediction"]["close"], entries[20]["date"]) == (990.0, "2026-10-29")
    assert (entries[5]["prediction"], entries[5]["models"]) == (None, [])


def test_nothing_predicted_yet(client: Client, db: None) -> None:
    add_stock()

    data = envelope_of(client.get("/api/stocks/2330/predictions"))["data"]

    assert [entry["date"] for entry in data["horizons"]] == [None, None]
    assert [entry["prediction"] for entry in data["horizons"]] == [None, None]
    assert [entry["universe"] for entry in data["horizons"]] == [0, 0]


def test_an_unknown_stock_is_not_found(client: Client, runs: Runs) -> None:
    response = client.get("/api/stocks/0000/predictions")

    assert response.status_code == 404
    assert envelope_of(response)["error"] == {"code": "not_found", "message": "找不到股票 0000。"}


def test_a_bad_date_is_invalid(client: Client, runs: Runs) -> None:
    response = client.get("/api/stocks/2330/predictions", {"date": "10/30"})

    assert response.status_code == 400
    assert set(envelope_of(response)["error"]["fields"]) == {"date"}


def test_stock_predictions_query_count_on_the_worst_path(
    client: Client, runs: Runs, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    Prediction.objects.filter(run=runs.vol, stock_id="6488").delete()
    add_predictions(runs.rank5, OCT_29, 5, {"6488": {"score": 0.1, "rank": 1, "pct": 1.0}})
    FeatureDaily.objects.create(stock_id="6488", date=OCT_30, in_universe=True, tech_vol_20d=0.5)

    with django_assert_max_num_queries(22):
        response = client.get("/api/stocks/6488/predictions")

    entries = horizons_of(envelope_of(response))
    assert [entries[h]["prediction"]["interval_source"] for h in (5, 20)] == ["tech_vol_20d"] * 2
    assert entries[5]["prediction"]["prev_rank"] == 1


def test_a_band_degraded_after_the_prediction_day_still_shows(client: Client, runs: Runs) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(
        status="degraded", params={"degraded_on": "2026-11-02"}
    )

    entries = horizons_of(envelope_of(client.get("/api/stocks/2330/predictions")))

    assert entries[20]["degraded"] == ["return", "interval"]
    assert entries[20]["prediction"]["interval_source"] == "quantile"
