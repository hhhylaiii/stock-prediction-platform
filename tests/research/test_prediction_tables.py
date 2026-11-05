import math
from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd
import pytest
from django.db import IntegrityError, connection, transaction
from django.db.models import Model, ProtectedError

from apps.etl.upsert import upsert
from apps.research.models import REST_FEATURE, ModelRun, Prediction, PredictionContrib
from tests.market.schema import index_columns, primary_key_columns

pytestmark = pytest.mark.django_db
DAY = date(2026, 10, 30)
RUN_ID = "prod-rank.lightgbm.h20"
SPEC_COLUMNS = {
    "run",
    "stock_id",
    "date",
    "horizon",
    "score",
    "rank",
    "pct",
    "ret_pred",
    "q10",
    "q50",
    "q90",
    "vol_pred",
    "prob_up_5pct",
    "in_universe",
}


@pytest.fixture
def model_run() -> ModelRun:
    return ModelRun.objects.create(
        run_id=RUN_ID,
        run_key="prod-rank",
        model_type="lightgbm",
        objective="rank",
        version="2026-10",
        train_start=date(2023, 9, 1),
        train_end=date(2026, 9, 30),
        lib_versions={"lightgbm": "4.5.0"},
        feature_hash="0" * 64,
    )


def prediction(**changes: Any) -> Prediction:
    fields: dict[str, Any] = {
        "run_id": RUN_ID,
        "stock_id": "2330",
        "date": DAY,
        "horizon": 20,
        "score": 0.61,
        "rank": 12,
        "pct": 0.994,
        "ret_pred": 0.012,
        "q10": -0.06,
        "q50": 0.012,
        "q90": 0.08,
        "vol_pred": 0.31,
        "prob_up_5pct": 0.27,
        "in_universe": True,
    }
    return Prediction(**{**fields, **changes})


def contrib(**changes: Any) -> PredictionContrib:
    fields: dict[str, Any] = {
        "run_id": RUN_ID,
        "stock_id": "2330",
        "date": DAY,
        "horizon": 20,
        "feature": "chip_fi_net_20d",
        "position": 1,
        "contrib": 0.031,
        "n_features": 1,
    }
    return PredictionContrib(**{**fields, **changes})


def test_prediction_keeps_every_spec_column() -> None:
    assert {field.name for field in Prediction._meta.get_fields()} >= SPEC_COLUMNS


def test_prediction_keys_and_indexes() -> None:
    assert primary_key_columns("prediction") == [["run_id", "stock_id", "date", "horizon"]]
    assert ["date", "horizon", "rank"] in index_columns("prediction")
    assert ["stock_id", "date"] in index_columns("prediction")


def test_contrib_keys_and_indexes() -> None:
    assert primary_key_columns("prediction_contrib") == [
        ["run_id", "stock_id", "date", "horizon", "position"]
    ]
    assert ["stock_id", "date"] in index_columns("prediction_contrib")


def test_prediction_round_trip_keeps_nulls_for_outputs_of_other_runs(model_run: ModelRun) -> None:
    prediction().save()
    prediction(stock_id="9999", score=0.1, rank=None, pct=None, in_universe=False).save()
    prediction(
        stock_id="1101",
        horizon=5,
        q10=None,
        q50=None,
        q90=None,
        vol_pred=None,
        prob_up_5pct=None,
    ).save()

    stored = Prediction.objects.get(pk=(RUN_ID, "2330", DAY, 20))
    outside = Prediction.objects.get(pk=(RUN_ID, "9999", DAY, 20))
    five_day = Prediction.objects.get(pk=(RUN_ID, "1101", DAY, 5))
    assert (stored.q10, stored.q50, stored.q90) == (-0.06, 0.012, 0.08)
    assert stored.run.objective == "rank"
    assert (outside.rank, outside.pct, outside.in_universe) == (None, None, False)
    assert five_day.prob_up_5pct is None


def test_prediction_upsert_is_idempotent(model_run: ModelRun) -> None:
    frame = pd.DataFrame(
        [
            {"run_id": RUN_ID, "stock_id": stock, "date": DAY, "horizon": 20, "score": score}
            | {"rank": rank, "pct": 1 - (rank - 1) / 2, "in_universe": True}
            for stock, score, rank in [("2330", 0.6, 1), ("2317", 0.4, 2)]
        ]
    )
    keys = ["run_id", "stock_id", "date", "horizon"]
    values = ["score", "rank", "pct", "in_universe"]

    upsert(Prediction, frame, keys, values)
    upsert(Prediction, frame.assign(score=[0.7, 0.3]), keys, values)
    upsert(Prediction, frame.assign(score=[0.7, 0.3]), keys, values)

    rows = Prediction.objects.order_by("rank").values_list("stock_id", "score")
    assert list(rows) == [("2330", 0.7), ("2317", 0.3)]


@pytest.mark.parametrize(
    ("changes", "constraint"),
    [
        ({"horizon": 10}, "prediction_horizon_known"),
        ({"score": math.nan}, "prediction_finite"),
        ({"q90": math.inf}, "prediction_finite"),
        ({"rank": 0}, "prediction_rank_gte_1"),
        ({"pct": 1.01}, "prediction_unit_interval"),
        ({"pct": -0.01}, "prediction_unit_interval"),
        ({"prob_up_5pct": 1.2}, "prediction_unit_interval"),
        ({"vol_pred": -0.01}, "prediction_vol_gte_0"),
        ({"q10": 0.02}, "prediction_quantiles_ordered"),
        ({"q90": 0.0}, "prediction_quantiles_ordered"),
        ({"q10": 0.09, "q50": None}, "prediction_quantiles_ordered"),
        ({"score": None}, "prediction_rank_pct_with_score"),
        ({"pct": None}, "prediction_rank_pct_with_score"),
        ({"rank": None}, "prediction_rank_pct_with_score"),
        ({"ret_pred": 0.02}, "prediction_ret_pred_is_q50"),
    ],
)
def test_prediction_check_constraints(
    model_run: ModelRun, changes: dict[str, Any], constraint: str
) -> None:
    with pytest.raises(IntegrityError, match=constraint), transaction.atomic():
        prediction(**changes).save()


OUTPUTS = [
    (prediction, Prediction, "prediction"),
    (contrib, PredictionContrib, "prediction_contrib"),
]


@pytest.mark.parametrize(("build", "model", "table"), OUTPUTS)
def test_outputs_need_a_recorded_run(
    build: Callable[..., Model], model: type[Model], table: str
) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")

    with pytest.raises(IntegrityError, match=f"{table}_run_id"), transaction.atomic():
        build(run_id="never-trained").save()


@pytest.mark.parametrize(("build", "model", "table"), OUTPUTS)
def test_a_run_with_outputs_cannot_be_deleted(
    model_run: ModelRun, build: Callable[..., Model], model: type[Model], table: str
) -> None:
    build().save()

    with pytest.raises(ProtectedError), transaction.atomic():
        model_run.delete()

    assert model._default_manager.count() == 1


def test_contrib_keeps_top_features_and_one_rest_row(model_run: ModelRun) -> None:
    contrib().save()
    contrib(feature="fund_rev_yoy", position=2, contrib=-0.012).save()
    contrib(feature=REST_FEATURE, position=3, contrib=0.004, n_features=38).save()

    rows = PredictionContrib.objects.filter(run_id=RUN_ID, stock_id="2330", date=DAY, horizon=20)
    assert list(rows.order_by("position").values_list("feature", "n_features")) == [
        ("chip_fi_net_20d", 1),
        ("fund_rev_yoy", 1),
        (REST_FEATURE, 38),
    ]


@pytest.mark.parametrize(
    ("changes", "constraint"),
    [
        ({"horizon": 60}, "prediction_contrib_horizon_known"),
        ({"position": 0}, "prediction_contrib_position_gte_1"),
        ({"contrib": math.nan}, "prediction_contrib_finite"),
        ({"contrib": -math.inf}, "prediction_contrib_finite"),
        ({"n_features": 0}, "prediction_contrib_rest_sums_features"),
        ({"n_features": 3}, "prediction_contrib_rest_sums_features"),
        ({"feature": REST_FEATURE, "n_features": 0}, "prediction_contrib_rest_sums_features"),
    ],
)
def test_contrib_check_constraints(
    model_run: ModelRun, changes: dict[str, Any], constraint: str
) -> None:
    with pytest.raises(IntegrityError, match=constraint), transaction.atomic():
        contrib(**changes).save()


def contrib_frame(features: list[tuple[str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"run_id": RUN_ID, "stock_id": "2330", "date": DAY, "horizon": 20}
            | {"position": position, "feature": name, "contrib": value, "n_features": count}
            for position, (name, value, count) in enumerate(features, start=1)
        ]
    )


def test_contrib_upsert_rewrites_a_reranked_prediction(model_run: ModelRun) -> None:
    keys = ["run_id", "stock_id", "date", "horizon", "position"]
    values = ["feature", "contrib", "n_features"]
    first = [("tech_ret_20d", 0.03, 1), ("fund_rev_yoy", -0.02, 1), ("chip_fi_net_5d", 0.01, 1)]
    reranked = [("fund_rev_yoy", 0.04, 1), ("tech_ret_20d", 0.02, 1), ("size_log_mcap", 0.01, 1)]
    rest = [(REST_FEATURE, 0.002, 38)]

    upsert(PredictionContrib, contrib_frame(first + rest), keys, values)
    upsert(PredictionContrib, contrib_frame(reranked + rest), keys, values)
    upsert(PredictionContrib, contrib_frame(reranked + rest), keys, values)
    connection.check_constraints()

    rows = PredictionContrib.objects.order_by("position").values_list("feature", "contrib")
    assert list(rows) == [(name, value) for name, value, _ in reranked + rest]


def save_and_check(row: Model) -> None:
    row.save()
    connection.check_constraints()


def test_contrib_features_are_unique_per_prediction(model_run: ModelRun) -> None:
    contrib().save()
    save_and_check(contrib(stock_id="2317"))

    with (
        pytest.raises(IntegrityError, match="prediction_contrib_feature_unique"),
        transaction.atomic(),
    ):
        save_and_check(contrib(position=2))
