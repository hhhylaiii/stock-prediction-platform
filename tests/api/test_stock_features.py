from datetime import date
from typing import Any

import pytest
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.research.models import REST_FEATURE, FeatureDaily, ModelRun, PredictionContrib
from tests.api.conftest import (
    OCT_29,
    OCT_30,
    Runs,
    add_model_run,
    add_predictions,
    add_stock,
    envelope_of,
)

pytestmark = pytest.mark.usefixtures("calendar_in_db")

TOP = (
    ("tech_ret_60d", 0.031),
    ("chip_fi_net_20d", -0.024),
    ("tech_parkinson_vol_20d", 0.012),
    ("size_log_mcap", 0.008),
    ("fund_roe_ttm", -0.005),
)


def add_contribs(
    run: ModelRun,
    day: date,
    horizon: int,
    top: tuple[tuple[str, float], ...],
    rest: tuple[float, int] | None = (0.004, 72),
    stock_id: str = "2330",
) -> None:
    rows = [
        PredictionContrib(
            run=run,
            stock_id=stock_id,
            date=day,
            horizon=horizon,
            feature=feature,
            position=position,
            contrib=value,
        )
        for position, (feature, value) in enumerate(top, start=1)
    ]
    if rest is not None:
        rows.append(
            PredictionContrib(
                run=run,
                stock_id=stock_id,
                date=day,
                horizon=horizon,
                feature=REST_FEATURE,
                position=len(top) + 1,
                contrib=rest[0],
                n_features=rest[1],
            )
        )
    PredictionContrib.objects.bulk_create(rows)


@pytest.fixture
def contribs(runs: Runs) -> Runs:
    add_contribs(runs.rank20, OCT_30, 20, TOP)
    add_contribs(runs.rank20, OCT_29, 20, (("tech_ret_5d", 0.5),), rest=None)
    add_contribs(runs.rank5, OCT_30, 5, (("tech_ret_1d", 0.02), ("tech_rsi_14", -0.01)), (0.0, 75))
    add_contribs(runs.rank20, OCT_30, 20, (("tech_ret_1d", 0.9),), stock_id="2317")
    FeatureDaily.objects.create(
        stock_id="2330", date=OCT_29, in_universe=True, tech_ret_60d=0.99, tech_ret_5d=0.07
    )
    FeatureDaily.objects.create(
        stock_id="2330",
        date=OCT_30,
        in_universe=True,
        tech_ret_60d=0.12,
        tech_ret_5d=0.5,
        size_log_mcap=30.5,
        fund_roe_ttm=None,
    )
    return runs


def test_top_contributions_come_in_order_with_the_rest(client: Client, contribs: Runs) -> None:
    response = client.get("/api/stocks/2330/features")

    assert response.status_code == 200
    body = envelope_of(response)
    items: list[dict[str, Any]] = body["data"]["items"]
    assert [(item["position"], item["feature"], item["contrib"]) for item in items] == [
        (n, feature, value) for n, (feature, value) in enumerate(TOP, start=1)
    ]
    assert items[0] == {
        "position": 1,
        "feature": "tech_ret_60d",
        "label": "60 日報酬",
        "unit": "ratio",
        "group": "technical",
        "contrib": 0.031,
        "value": 0.12,
    }
    assert [(item["label"], item["unit"]) for item in items[1:]] == [
        ("外資 20 日買賣超（占流通股）", "ratio"),
        ("20 日 Parkinson 波動（年化）", "percent"),
        ("市值（對數）", "number"),
        ("ROE（近四季）", "percent"),
    ]
    assert [item["value"] for item in items] == [0.12, None, None, 30.5, None]
    assert items[2]["group"] == "technical"
    assert body["data"]["rest"] == {"contrib": 0.004, "n_features": 72}
    assert body["data"]["n_features"] == 77
    assert body["data"]["total"] == pytest.approx(sum(value for _, value in TOP) + 0.004)
    assert body["meta"] == {
        "date": "2026-10-30",
        "horizon": 20,
        "run_id": "rank-h20",
        "model_type": "lightgbm",
        "version": "2026-10",
    }


def test_five_day_contributions_come_from_the_five_day_rank_run(
    client: Client, contribs: Runs
) -> None:
    body = envelope_of(client.get("/api/stocks/2330/features", {"horizon": "5"}))

    assert [item["feature"] for item in body["data"]["items"]] == ["tech_ret_1d", "tech_rsi_14"]
    assert (body["data"]["n_features"], body["meta"]["run_id"]) == (77, "rank-h5")


def test_only_the_serving_run_s_contributions_count(client: Client, contribs: Runs) -> None:
    for run_id, status in (("rank-old", "retired"), ("rank-cand", "candidate")):
        other = add_model_run(run_id, "rank", status=status)
        add_predictions(other, OCT_30, 20, {"2330": {"score": 0.0, "rank": 9, "pct": 0.0}})
        add_contribs(other, OCT_30, 20, (("mkt_taiex_ret_5d", 0.7),))

    body = envelope_of(client.get("/api/stocks/2330/features"))

    assert body["meta"]["run_id"] == "rank-h20"
    assert "mkt_taiex_ret_5d" not in [item["feature"] for item in body["data"]["items"]]


def test_an_unknown_feature_keeps_its_code(client: Client, runs: Runs) -> None:
    add_contribs(runs.rank20, OCT_30, 20, (("tech_future_x", 0.1),), rest=None)

    item = envelope_of(client.get("/api/stocks/2330/features"))["data"]["items"][0]

    assert (item["label"], item["unit"], item["group"]) == ("tech_future_x", None, None)


def test_an_earlier_date_without_a_rest_row(client: Client, contribs: Runs) -> None:
    data = envelope_of(client.get("/api/stocks/2330/features", {"date": "2026-10-29"}))["data"]

    assert [(item["feature"], item["value"]) for item in data["items"]] == [("tech_ret_5d", 0.07)]
    assert (data["rest"], data["n_features"], data["total"]) == (None, 1, 0.5)


def test_no_contributions_is_an_empty_answer(client: Client, runs: Runs) -> None:
    body = envelope_of(client.get("/api/stocks/6488/features"))

    assert body["data"] == {"items": [], "rest": None, "total": None, "n_features": 0}
    assert body["meta"]["run_id"] == "rank-h20"


def test_nothing_served_yet(client: Client, db: None) -> None:
    add_stock()

    body = envelope_of(client.get("/api/stocks/2330/features"))

    assert body["data"]["items"] == []
    assert body["meta"] == {
        "date": None,
        "horizon": 20,
        "run_id": None,
        "model_type": None,
        "version": None,
    }


def test_an_unknown_stock_is_not_found(client: Client, contribs: Runs) -> None:
    response = client.get("/api/stocks/0000/features")

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("params", "field"), [({"horizon": "10"}, "horizon"), ({"date": "x"}, "date")]
)
def test_bad_queries_are_invalid(
    client: Client, contribs: Runs, params: dict[str, str], field: str
) -> None:
    response = client.get("/api/stocks/2330/features", params)

    assert response.status_code == 400
    assert set(envelope_of(response)["error"]["fields"]) == {field}


def test_features_query_count(
    client: Client, contribs: Runs, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    with django_assert_max_num_queries(6):
        client.get("/api/stocks/2330/features")
