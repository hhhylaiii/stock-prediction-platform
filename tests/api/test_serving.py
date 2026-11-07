from datetime import date

import pytest
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.api.model_runs import PAGE_ORDER
from apps.api.serving import Slot, latest_days, latest_served_date, served_monitor, warn_unplaced
from apps.research.models import ModelMonitor, ModelRun
from apps.research.runs import SCREENING_MODEL
from tests.api.conftest import (
    SESSIONS,
    add_model_run,
    add_predictions,
    add_stock,
    envelope_of,
    seed_market,
)

pytestmark = pytest.mark.django_db

OCT_29, OCT_30, NOV_2 = date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2)
RANKED = {"2330": {"score": 0.5, "rank": 1, "pct": 0.5}}
RANK20, RANK5 = Slot("rank", 20), Slot("rank", 5)
TRAINED = date(2026, 9, 30)


def test_each_slot_takes_the_newest_day_of_its_active_and_degraded_runs() -> None:
    add_stock()
    add_predictions(add_model_run("rank-a", "rank"), OCT_29, 20, RANKED)
    add_predictions(add_model_run("rank-b", "rank", status="degraded"), OCT_30, 20, RANKED)
    add_predictions(add_model_run("rank-g", "rank"), OCT_29, 20, RANKED)
    add_predictions(add_model_run("rank-c", "rank", status="retired"), NOV_2, 20, RANKED)
    add_predictions(add_model_run("rank-d", "rank", status="candidate"), NOV_2, 20, RANKED)
    add_predictions(add_model_run("rank-e", "rank", model_type=SCREENING_MODEL), NOV_2, 20, RANKED)
    add_predictions(add_model_run("rank-f", "rank"), OCT_29, 5, RANKED)
    add_predictions(add_model_run("vol-a", "volatility"), NOV_2, 20, {"2330": {"vol_pred": 0.3}})
    add_predictions(add_model_run("reg-a", "regression"), NOV_2, 20, {"2330": {"ret_pred": 0.01}})

    days = latest_days([RANK20, RANK5, Slot("regression", 5)])

    assert days == {RANK20: OCT_30, RANK5: OCT_29}
    assert (latest_served_date(20), latest_served_date(5)) == (OCT_30, OCT_29)


def test_no_served_run_leaves_every_slot_out() -> None:
    add_stock()
    add_predictions(add_model_run("rank-c", "rank", status="retired"), NOV_2, 20, RANKED)

    assert latest_days(PAGE_ORDER) == {}
    assert latest_served_date(20) is None


def test_the_lookup_is_one_query_for_every_slot(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    add_stock()
    for n in range(4):
        add_predictions(add_model_run(f"rank-{n}", "rank"), OCT_30, 20, RANKED)

    with django_assert_num_queries(1):
        latest_days(PAGE_ORDER)


def sessions_after(day: date, count: int) -> list[date]:
    start = SESSIONS.index(day) + 1
    return SESSIONS[start : start + count]


def monitor(run: ModelRun, horizon: int, days: list[date], ic: float = 0.05) -> None:
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=run, date=day, horizon=horizon, ic=ic, n_stocks=3) for day in days
    )


def served_days(rows: list[dict[str, object]]) -> list[object]:
    return [row["date"] for row in rows]


@pytest.mark.usefixtures("calendar_in_db")
def test_days_whose_label_window_reaches_into_training_are_not_out_of_sample() -> None:
    run = add_model_run("rank-a", "rank", train_end=TRAINED)
    days = sessions_after(TRAINED, 25)
    monitor(run, 20, days)

    served = served_monitor({RANK20: (TRAINED, days[-1])})

    assert served_days(served[RANK20]) == days[19:]


@pytest.mark.usefixtures("calendar_in_db")
def test_each_horizon_of_a_run_starts_out_of_sample_on_its_own_day() -> None:
    run = add_model_run("rank-a", "rank", train_end=TRAINED)
    days = sessions_after(TRAINED, 25)
    monitor(run, 5, days)
    monitor(run, 20, days)

    served = served_monitor({RANK5: (TRAINED, days[-1]), RANK20: (TRAINED, days[-1])})

    assert served_days(served[RANK5]) == days[4:]
    assert served_days(served[RANK20]) == days[19:]


def unplaced(run_id: str, reason: str) -> str:
    return (
        f"run {run_id} has no 20-day out-of-sample day in the calendar ({reason});"
        " its rows are left out"
    )


def warnings_in(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]


@pytest.mark.usefixtures("calendar_in_db")
def test_a_boundary_past_the_calendar_leaves_only_that_run_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    warn_unplaced.cache_clear()
    recent = SESSIONS[-10]
    old = add_model_run("rank-old", "rank", status="retired", train_end=TRAINED)
    new = add_model_run("rank-new", "rank", train_end=recent)
    days = SESSIONS[-9:]
    monitor(old, 20, days, ic=0.02)
    monitor(new, 20, days, ic=0.9)

    served = served_monitor({RANK20: (recent, days[-1])})

    assert served_days(served[RANK20]) == days
    assert {row["run_id"] for row in served[RANK20]} == {"rank-old"}
    assert warnings_in(caplog) == [
        unplaced("rank-new", f"shifting {recent} by 20 leaves the calendar")
    ]


@pytest.mark.usefixtures("calendar_in_db")
def test_a_train_end_off_the_sessions_leaves_the_run_out_with_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    warn_unplaced.cache_clear()
    saturday = date(2026, 10, 3)
    monitor(add_model_run("rank-a", "rank", train_end=saturday), 20, SESSIONS[-9:])
    window = {RANK20: (saturday, SESSIONS[-1])}

    assert served_monitor(window) == served_monitor(window) == {}
    assert warnings_in(caplog) == [unplaced("rank-a", f"{saturday} is not a trading day")]


def test_an_unreadable_calendar_counts_no_monitor_row_and_logs_why(
    caplog: pytest.LogCaptureFixture,
) -> None:
    monitor(add_model_run("rank-a", "rank", train_end=TRAINED), 20, sessions_after(TRAINED, 25))

    assert served_monitor({RANK20: (TRAINED, SESSIONS[-1])}) == {}
    assert [r.getMessage() for r in caplog.records if r.levelname == "ERROR"] == [
        "trading calendar cannot be read (trading_calendar is empty);"
        " no monitor row counts as out of sample"
    ]


@pytest.mark.usefixtures("calendar_in_db")
def test_the_models_page_still_answers_when_a_boundary_is_past_the_calendar(
    client: Client,
) -> None:
    seed_market()
    recent = SESSIONS[-10]
    run = add_model_run("rank-new", "rank", train_end=recent)
    add_predictions(run, SESSIONS[-1], 20, RANKED)
    monitor(run, 20, SESSIONS[-9:], ic=0.9)

    response = client.get("/api/models")

    assert response.status_code == 200
    rank = next(
        entry
        for entry in envelope_of(response)["data"]
        if (entry["objective"], entry["horizon"]) == ("rank", 20)
    )
    assert rank["recent"]["days"] == 0
