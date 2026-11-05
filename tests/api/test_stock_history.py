import math
from datetime import date
from typing import Any

import pytest
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.api.serving import warn_undated
from apps.research.models import FeatureDaily, LabelDaily, ModelRun
from tests.api.conftest import (
    SESSIONS,
    add_model_run,
    add_predictions,
    add_stock,
    envelope_of,
    taipei,
)

pytestmark = pytest.mark.usefixtures("calendar_in_db")

LAST_LABELLED = date(2026, 10, 1)
LABELLED = [day for day in SESSIONS if date(2026, 7, 6) <= day <= LAST_LABELLED]
PREDICTED = [day for day in SESSIONS if date(2026, 7, 6) <= day <= date(2026, 10, 30)]
QUANTILES = {"q10": -0.05, "q50": 0.01, "q90": 0.08, "ret_pred": 0.01}
TRAINED_TO = date(2026, 6, 30)


def serving(run_id: str, objective: str, **fields: Any) -> ModelRun:
    """A run trained before every predicted day, so all of them are out of sample."""
    return add_model_run(run_id, objective, train_end=TRAINED_TO, **fields)


def picks(days: list[date], step: int) -> list[date]:
    """Expected non-overlapping dates when every session in days has a prediction and a label."""
    index = SESSIONS.index(days[-1])
    chosen = []
    while index >= 0 and SESSIONS[index] >= days[0]:
        chosen.append(SESSIONS[index])
        index -= step
    return chosen


def add_labels(horizon: int, actual: dict[date, float | None], stock_id: str = "2330") -> None:
    """fwd_ret is set apart from fwd_ret_open so reading the wrong label shows."""
    LabelDaily.objects.bulk_create(
        LabelDaily(
            stock_id=stock_id,
            date=day,
            horizon=horizon,
            fwd_ret=None if value is None else value + 0.5,
            fwd_ret_open=value,
        )
        for day, value in actual.items()
    )


def actual_on(day: date) -> float:
    """Above q90 on the second pick, below q10 on the third, inside otherwise."""
    expected = picks(LABELLED, 20)
    return {expected[1]: 0.2, expected[2]: -0.2}.get(day, 0.03)


def outside_on(day: date) -> bool:
    return actual_on(day) != 0.03


@pytest.fixture
def history(db: None) -> ModelRun:
    add_stock()
    add_stock(stock_id="2317", name="鴻海")
    quantile = serving("quantile-h20", "quantile")
    other = {"q10": -0.6, "q50": -0.4, "q90": -0.3, "ret_pred": -0.4}
    for day in PREDICTED:
        add_predictions(quantile, day, 20, {"2330": QUANTILES, "2317": other})
    add_labels(20, {day: actual_on(day) for day in LABELLED})
    add_labels(20, dict.fromkeys(LABELLED, -0.9), stock_id="2317")
    return quantile


def rows_of(body: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = body["data"]
    return rows


def test_matured_twenty_day_predictions_every_twenty_sessions(
    client: Client, history: ModelRun
) -> None:
    response = client.get("/api/stocks/2330/history", {"page_size": "100"})

    assert response.status_code == 200
    body = envelope_of(response)
    expected = picks(LABELLED, 20)
    assert [row["date"] for row in rows_of(body)] == [day.isoformat() for day in expected]
    assert rows_of(body)[0] == {
        "date": LAST_LABELLED.isoformat(),
        "ret_pred": 0.01,
        "actual": 0.03,
        "interval": {"ret_low": -0.05, "ret_high": 0.08},
        "interval_source": "quantile",
        "inside": True,
        "in_universe": True,
        "run_id": "quantile-h20",
    }
    inside = sum(not outside_on(day) for day in expected)
    assert body["meta"]["horizon"] == 20
    assert (body["meta"]["total"], body["meta"]["inside"], body["meta"]["with_interval"]) == (
        len(expected),
        inside,
        len(expected),
    )
    assert [row["inside"] for row in rows_of(body)] == [not outside_on(day) for day in expected]


def test_a_missing_prediction_moves_the_next_pick_back(client: Client, history: ModelRun) -> None:
    ModelRun.objects.get(pk="quantile-h20").predictions.filter(date=LAST_LABELLED).delete()

    rows = rows_of(envelope_of(client.get("/api/stocks/2330/history")))

    previous = SESSIONS[SESSIONS.index(LAST_LABELLED) - 1]
    assert rows[0]["date"] == previous.isoformat()
    assert rows[1]["date"] == SESSIONS[SESSIONS.index(previous) - 20].isoformat()


def test_each_date_uses_the_run_that_served_it(client: Client, history: ModelRun) -> None:
    old = serving("quantile-old", "quantile", status="retired")
    newer = serving("quantile-cand", "quantile", status="candidate")
    wide = {"q10": -0.5, "q50": 0.3, "q90": 0.9, "ret_pred": 0.3}
    add_predictions(old, LAST_LABELLED, 20, {"2330": wide})
    add_predictions(newer, LAST_LABELLED, 20, {"2330": wide})
    first = picks(LABELLED, 20)[-1]
    ModelRun.objects.get(pk="quantile-h20").predictions.filter(date=first).delete()
    add_predictions(old, first, 20, {"2330": wide})

    rows = rows_of(envelope_of(client.get("/api/stocks/2330/history", {"page_size": "100"})))

    assert (rows[0]["run_id"], rows[0]["ret_pred"]) == ("quantile-h20", 0.01)
    assert (rows[-1]["date"], rows[-1]["run_id"], rows[-1]["ret_pred"]) == (
        first.isoformat(),
        "quantile-old",
        0.3,
    )


def test_a_degraded_quantile_run_is_judged_on_its_volatility_band(
    client: Client, history: ModelRun, caplog: pytest.LogCaptureFixture
) -> None:
    warn_undated.cache_clear()
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")
    vol = serving("vol-h20", "volatility")
    add_predictions(vol, LAST_LABELLED, 20, {"2330": {"vol_pred": 0.3}})
    spread = 1.28 * 0.3 * math.sqrt(20 / 252)

    rows = rows_of(envelope_of(client.get("/api/stocks/2330/history")))

    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == [
        "degraded run quantile-h20 has no params.degraded_on; its band is set aside on every day"
    ]
    assert rows[0]["interval_source"] == "volatility"
    assert rows[0]["interval"] == {
        "ret_low": round(0.01 - spread, 6),
        "ret_high": round(0.01 + spread, 6),
    }
    assert (rows[1]["interval"], rows[1]["interval_source"], rows[1]["inside"]) == (
        None,
        None,
        None,
    )


def test_five_day_history_steps_five_sessions_on_the_regression(
    client: Client, history: ModelRun
) -> None:
    regression = serving("reg-h5", "regression")
    vol = serving("vol-h20", "volatility", created_at=taipei(2026, 6, 1))
    for day in PREDICTED:
        add_predictions(regression, day, 5, {"2330": {"ret_pred": -0.01}})
        add_predictions(vol, day, 20, {"2330": {"vol_pred": 0.2}})
    add_labels(5, dict.fromkeys(LABELLED, 0.0))

    body = envelope_of(client.get("/api/stocks/2330/history", {"horizon": "5", "page_size": "5"}))
    spread = 1.28 * 0.2 * math.sqrt(5 / 252)

    assert [row["date"] for row in rows_of(body)] == [
        day.isoformat() for day in picks(LABELLED, 5)[:5]
    ]
    assert rows_of(body)[0]["interval"]["ret_high"] == round(-0.01 + spread, 6)
    assert rows_of(body)[0]["run_id"] == "reg-h5"
    assert body["meta"]["total"] == len(picks(LABELLED, 5))


def test_without_vol_pred_the_band_uses_that_day_s_tech_vol_20d(
    client: Client, history: ModelRun
) -> None:
    regression = serving("reg-h5", "regression")
    for day in PREDICTED:
        add_predictions(regression, day, 5, {"2330": {"ret_pred": 0.0}})
    add_labels(5, dict.fromkeys(LABELLED, 0.0))
    FeatureDaily.objects.create(
        stock_id="2330", date=LAST_LABELLED, in_universe=True, tech_vol_20d=0.4
    )
    second = picks(LABELLED, 5)[1]
    FeatureDaily.objects.create(stock_id="2317", date=second, in_universe=True, tech_vol_20d=0.4)
    spread = 1.28 * 0.4 * math.sqrt(5 / 252)

    rows = rows_of(envelope_of(client.get("/api/stocks/2330/history", {"horizon": "5"})))

    assert rows[0]["interval_source"] == "tech_vol_20d"
    assert rows[0]["interval"] == {"ret_low": round(-spread, 6), "ret_high": round(spread, 6)}
    assert rows[1]["interval"] is None


def test_days_outside_the_training_universe_are_flagged_and_still_counted(
    client: Client, history: ModelRun
) -> None:
    history.predictions.filter(stock_id="2330", date=LAST_LABELLED).update(in_universe=False)

    first = envelope_of(client.get("/api/stocks/2330/history", {"page_size": "1"}))
    rows = rows_of(envelope_of(client.get("/api/stocks/2330/history")))

    expected = picks(LABELLED, 20)
    assert [row["in_universe"] for row in rows[:2]] == [False, True]
    assert len(first["data"]) == 1
    assert (first["meta"]["inside"], first["meta"]["with_interval"]) == (
        sum(not outside_on(day) for day in expected),
        len(expected),
    )


def test_a_return_on_either_end_of_the_band_is_inside(client: Client, history: ModelRun) -> None:
    expected = picks(LABELLED, 20)
    LabelDaily.objects.filter(stock_id="2330", horizon=20, date=expected[1]).update(
        fwd_ret_open=0.08
    )
    LabelDaily.objects.filter(stock_id="2330", horizon=20, date=expected[2]).update(
        fwd_ret_open=-0.05
    )

    rows = rows_of(envelope_of(client.get("/api/stocks/2330/history")))

    assert [row["inside"] for row in rows[1:3]] == [True, True]


def test_a_gap_inside_a_window_does_not_shift_the_picks(client: Client, history: ModelRun) -> None:
    expected = picks(LABELLED, 20)
    hole = SESSIONS[SESSIONS.index(expected[0]) - 5]
    history.predictions.filter(date=hole).delete()

    rows = rows_of(envelope_of(client.get("/api/stocks/2330/history")))

    assert [row["date"] for row in rows] == [day.isoformat() for day in expected]


def test_days_inside_the_run_s_training_window_are_not_history(
    client: Client, history: ModelRun
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(train_end=date(2026, 9, 15))

    rows = rows_of(envelope_of(client.get("/api/stocks/2330/history")))

    assert [row["date"] for row in rows] == [LAST_LABELLED.isoformat()]


def test_unlabelled_or_null_labels_are_not_matured(client: Client, history: ModelRun) -> None:
    LabelDaily.objects.filter(date=LAST_LABELLED).update(fwd_ret_open=None)

    rows = rows_of(envelope_of(client.get("/api/stocks/2330/history")))

    assert rows[0]["date"] == SESSIONS[SESSIONS.index(LAST_LABELLED) - 1].isoformat()


def test_no_history_yet(client: Client, db: None) -> None:
    add_stock()

    body = envelope_of(client.get("/api/stocks/2330/history"))

    assert body["data"] == []
    assert (body["meta"]["total"], body["meta"]["inside"], body["meta"]["with_interval"]) == (
        0,
        0,
        0,
    )


def test_an_unknown_stock_is_not_found(client: Client, history: ModelRun) -> None:
    assert client.get("/api/stocks/0000/history").status_code == 404


def test_a_bad_horizon_is_invalid(client: Client, history: ModelRun) -> None:
    response = client.get("/api/stocks/2330/history", {"horizon": "10"})

    assert response.status_code == 400
    assert set(envelope_of(response)["error"]["fields"]) == {"horizon"}


def test_history_query_count(
    client: Client, history: ModelRun, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    with django_assert_max_num_queries(6):
        client.get("/api/stocks/2330/history")


@pytest.mark.parametrize("status", ["degraded", "retired"])
def test_the_band_is_set_aside_only_from_the_degraded_on_day(
    client: Client, history: ModelRun, status: str
) -> None:
    expected = picks(LABELLED, 20)
    ModelRun.objects.filter(pk="quantile-h20").update(
        status=status, params={"degraded_on": expected[1].isoformat()}
    )
    vol = serving("vol-h20", "volatility")
    for day in expected[:3]:
        add_predictions(vol, day, 20, {"2330": {"vol_pred": 0.3}})

    rows = rows_of(envelope_of(client.get("/api/stocks/2330/history")))

    assert [row["interval_source"] for row in rows[:3]] == ["volatility", "volatility", "quantile"]
    assert rows[2]["interval"] == {"ret_low": -0.05, "ret_high": 0.08}


def test_an_unreadable_degraded_on_falls_back_to_every_day(
    client: Client, history: ModelRun, caplog: pytest.LogCaptureFixture
) -> None:
    warn_undated.cache_clear()
    ModelRun.objects.filter(pk="quantile-h20").update(
        status="degraded", params={"degraded_on": "soon"}
    )

    rows = rows_of(envelope_of(client.get("/api/stocks/2330/history")))

    assert all(row["interval"] is None for row in rows)
    assert "quantile-h20" in caplog.text
