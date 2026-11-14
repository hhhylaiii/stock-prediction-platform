import time
from datetime import date
from typing import Any

import pytest
from django.db import connection
from django.test import Client
from django.test.utils import CaptureQueriesContext

from apps.api.params import ORDERINGS
from apps.research.models import Prediction
from tests.api.conftest import (
    EARLY_CALENDAR_START,
    OCT_29,
    OCT_30,
    Runs,
    add_bars,
    add_model_run,
    add_predictions,
    add_stock,
    envelope_of,
)

pytestmark = pytest.mark.usefixtures("calendar_in_db")


@pytest.fixture
def calendar_start() -> date:
    """Runs here train before July, so their out-of-sample days start inside the calendar."""
    return EARLY_CALENDAR_START


def codes_of(client: Client, **params: Any) -> list[str]:
    return [row["stock_id"] for row in envelope_of(client.get("/api/predictions", params))["data"]]


def test_the_default_order_is_the_rank_model_and_says_so(client: Client, runs: Runs) -> None:
    body = envelope_of(client.get("/api/predictions"))

    assert [row["stock_id"] for row in body["data"]] == ["2330", "6488", "2317", "8299"]
    assert body["meta"]["ordering"] == "rank"
    assert codes_of(client, ordering="rank") == ["2330", "6488", "2317", "8299"]
    assert codes_of(client, ordering="") == ["2330", "6488", "2317", "8299"]


@pytest.mark.parametrize(
    ("ordering", "codes"),
    [
        ("-close", ["2330", "8299", "6488", "2317"]),
        ("close", ["2317", "6488", "8299", "2330"]),
        ("-ret_pred", ["2330", "8299", "2317", "6488"]),
        ("ret_pred", ["2317", "8299", "2330", "6488"]),
        ("-price_pred", ["2330", "8299", "2317", "6488"]),
        ("price_pred", ["2317", "8299", "2330", "6488"]),
        ("-prob_up_5pct", ["2330", "6488", "2317", "8299"]),
        ("prob_up_5pct", ["2317", "6488", "2330", "8299"]),
        ("prev_rank", ["6488", "2330", "2317", "8299"]),
        ("-prev_rank", ["2317", "2330", "6488", "8299"]),
    ],
)
def test_each_column_orders_the_whole_market_with_blanks_last(
    client: Client, runs: Runs, ordering: str, codes: list[str]
) -> None:
    body = envelope_of(client.get("/api/predictions", {"ordering": ordering}))

    assert [row["stock_id"] for row in body["data"]] == codes
    assert body["meta"]["ordering"] == ordering


@pytest.mark.parametrize(
    ("ordering", "field"),
    [
        ("-close", "close"),
        ("-ret_pred", "ret_pred"),
        ("-price_pred", "price_pred"),
        ("-prob_up_5pct", "prob_up_5pct"),
    ],
)
def test_the_order_follows_the_values_the_rows_show(
    client: Client, runs: Runs, ordering: str, field: str
) -> None:
    rows = envelope_of(client.get("/api/predictions", {"ordering": ordering}))["data"]

    shown = [row[field] for row in rows if row[field] is not None]
    assert shown == sorted(shown, reverse=True)
    assert all(row[field] is None for row in rows[len(shown) :])


def test_the_order_runs_across_pages(client: Client, runs: Runs) -> None:
    first = envelope_of(client.get("/api/predictions", {"ordering": "-close", "page_size": "2"}))
    second = codes_of(client, ordering="-close", page_size=2, page=2)

    assert [row["stock_id"] for row in first["data"]] + second == ["2330", "8299", "6488", "2317"]
    assert (first["meta"]["total"], first["meta"]["pages"]) == (4, 2)


def test_a_stock_without_trades_today_is_ordered_by_its_last_close(
    client: Client, runs: Runs
) -> None:
    rows = envelope_of(client.get("/api/predictions", {"ordering": "-close"}))["data"]

    assert (rows[1]["stock_id"], rows[1]["close"], rows[1]["close_date"]) == (
        "8299",
        800.0,
        "2026-10-29",
    )


def test_the_twenty_day_return_falls_back_to_the_median_quantile(
    client: Client, runs: Runs
) -> None:
    Prediction.objects.filter(run=runs.quantile, stock_id="2317").update(
        ret_pred=None, q50=0.08, q90=0.1
    )

    rows = envelope_of(client.get("/api/predictions", {"ordering": "-ret_pred"}))["data"]

    assert [(row["stock_id"], row["ret_pred"]) for row in rows] == [
        ("2317", 0.08),
        ("2330", 0.04),
        ("8299", 0.0),
        ("6488", None),
    ]


def test_five_days_order_by_the_regression_return(client: Client, runs: Runs) -> None:
    Prediction.objects.filter(run=runs.reg5, stock_id="6488").update(ret_pred=0.05)

    assert codes_of(client, horizon=5, ordering="-ret_pred") == ["6488", "2317", "2330"]
    assert codes_of(client, horizon=5, ordering="price_pred") == ["2317", "6488", "2330"]


def test_equal_values_keep_the_rank_order(client: Client, runs: Runs) -> None:
    Prediction.objects.filter(run=runs.quantile).update(ret_pred=0.01, q50=0.01)
    add_predictions(runs.quantile, OCT_30, 20, {"6488": {"ret_pred": 0.01, "q50": 0.01}})

    assert codes_of(client, ordering="-ret_pred") == ["2330", "6488", "2317", "8299"]
    assert codes_of(client, ordering="ret_pred") == ["2330", "6488", "2317", "8299"]


def test_the_previous_rank_comes_from_the_run_that_served_the_previous_session(
    client: Client, runs: Runs
) -> None:
    Prediction.objects.filter(run=runs.rank20, date=OCT_29).delete()
    old = add_model_run("rank-old", "rank", status="retired")
    add_predictions(
        old,
        OCT_29,
        20,
        {
            "2317": {"score": 0.9, "rank": 1, "pct": 0.99},
            "8299": {"score": 0.5, "rank": 2, "pct": 0.5},
            "2330": {"score": 0.1, "rank": 3, "pct": 0.01},
        },
    )

    rows = envelope_of(client.get("/api/predictions", {"ordering": "prev_rank"}))["data"]

    assert [(row["stock_id"], row["prev_rank"]) for row in rows] == [
        ("2317", 1),
        ("8299", 2),
        ("2330", 3),
        ("6488", None),
    ]


@pytest.mark.parametrize(
    ("params", "codes"),
    [
        ({"market": "TPEx", "ordering": "-close"}, ["8299", "6488"]),
        ({"q": "23", "ordering": "close"}, ["2317", "2330"]),
    ],
)
def test_filters_apply_before_the_order(
    client: Client, runs: Runs, params: dict[str, str], codes: list[str]
) -> None:
    body = envelope_of(client.get("/api/predictions", params))

    assert [row["stock_id"] for row in body["data"]] == codes
    assert (body["meta"]["total"], body["meta"]["universe"]) == (2, 3)


@pytest.mark.parametrize(
    ("params", "codes"),
    [
        ({"horizon": "5", "ordering": "-prob_up_5pct"}, ["2317", "2330", "6488"]),
        ({"horizon": "5", "ordering": "prev_rank"}, ["2317", "2330", "6488"]),
    ],
)
def test_a_column_without_a_serving_source_keeps_the_rank_order(
    client: Client, runs: Runs, params: dict[str, str], codes: list[str]
) -> None:
    body = envelope_of(client.get("/api/predictions", params))

    assert [row["stock_id"] for row in body["data"]] == codes
    assert body["meta"]["ordering"] == "rank"


@pytest.mark.parametrize("ordering", ["-ret_pred", "price_pred"])
def test_no_serving_return_model_keeps_the_rank_order(
    client: Client, runs: Runs, ordering: str
) -> None:
    Prediction.objects.filter(run=runs.quantile).delete()

    body = envelope_of(client.get("/api/predictions", {"ordering": ordering}))

    assert [row["stock_id"] for row in body["data"]] == ["2330", "6488", "2317", "8299"]
    assert body["meta"]["ordering"] == "rank"


def test_no_predictions_yet_answers_the_rank_order(client: Client, db: None) -> None:
    body = envelope_of(client.get("/api/predictions", {"ordering": "-close"}))

    assert (body["data"], body["meta"]["ordering"], body["meta"]["orderable"]) == ([], "rank", [])


@pytest.mark.parametrize(
    ("horizon", "columns"),
    [
        ("20", ["close", "ret_pred", "price_pred", "prob_up_5pct", "prev_rank"]),
        ("5", ["close", "ret_pred", "price_pred"]),
    ],
)
def test_meta_lists_the_columns_that_can_be_ordered(
    client: Client, runs: Runs, horizon: str, columns: list[str]
) -> None:
    meta = envelope_of(client.get("/api/predictions", {"horizon": horizon}))["meta"]

    assert meta["orderable"] == columns


def test_a_column_without_a_serving_source_is_not_orderable(client: Client, runs: Runs) -> None:
    Prediction.objects.filter(run__in=[runs.quantile, runs.prob]).delete()
    Prediction.objects.filter(run=runs.rank20, date=OCT_29).delete()

    meta = envelope_of(client.get("/api/predictions", {"q": "nothing"}))["meta"]

    assert (meta["orderable"], meta["total"]) == (["close"], 0)


@pytest.mark.parametrize(
    "ordering", ["-rank", "score", "-score", "interval", "vol_pred", "-stock_id", "close,rank"]
)
def test_an_unknown_ordering_is_invalid(client: Client, runs: Runs, ordering: str) -> None:
    response = client.get("/api/predictions", {"ordering": ordering})

    assert response.status_code == 400
    assert set(envelope_of(response)["error"]["fields"]) == {"ordering"}


def test_the_whitelist_is_the_rank_and_both_directions_of_five_columns() -> None:
    columns = ("close", "ret_pred", "price_pred", "prob_up_5pct", "prev_rank")
    expected = ("rank", *(f"{sign}{c}" for c in columns for sign in ("-", "")))
    assert expected == ORDERINGS


def query_count(client: Client, params: dict[str, str]) -> int:
    with CaptureQueriesContext(connection) as queries:
        assert client.get("/api/predictions", params).status_code == 200
    return len(queries)


def test_an_order_adds_no_query(client: Client, runs: Runs) -> None:
    default = query_count(client, {})

    counts = {ordering: query_count(client, {"ordering": ordering}) for ordering in ORDERINGS}

    assert set(counts.values()) == {default}


def test_two_thousand_stocks_answer_every_order_within_half_a_second(
    client: Client, db: None
) -> None:
    codes = [f"{9000 + n}" for n in range(2000)]
    for code in codes:
        add_stock(stock_id=code, name=f"股{code}")
    for n, code in enumerate(codes):
        add_bars(code, [OCT_29, OCT_30], [100.0 + n, 101.0 + n])
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
        {
            code: {"q10": -0.1, "q50": n / 1e5, "q90": 0.1, "ret_pred": n / 1e5}
            for n, code in enumerate(codes)
        },
    )
    client.get("/api/predictions", {"ordering": "-close"})

    for ordering in ("-close", "-ret_pred", "-price_pred", "prev_rank"):
        started = time.perf_counter()
        response = client.get("/api/predictions", {"ordering": ordering})
        elapsed = time.perf_counter() - started

        body = envelope_of(response)
        assert (body["meta"]["ordering"], body["data"][0]["stock_id"]) == (ordering, "10999")
        assert elapsed < 0.5, ordering
