from datetime import date

import pytest
from pytest_django import DjangoAssertNumQueries

from apps.api.model_runs import PAGE_ORDER
from apps.api.serving import Slot, latest_days, latest_served_date
from apps.research.runs import SCREENING_MODEL
from tests.api.conftest import add_model_run, add_predictions, add_stock

pytestmark = pytest.mark.django_db

OCT_29, OCT_30, NOV_2 = date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2)
RANKED = {"2330": {"score": 0.5, "rank": 1, "pct": 0.5}}
RANK20, RANK5 = Slot("rank", 20), Slot("rank", 5)


def test_each_slot_takes_the_newest_day_of_its_active_and_degraded_runs() -> None:
    add_stock()
    add_predictions(add_model_run("rank-a", "rank"), OCT_29, 20, RANKED)
    add_predictions(add_model_run("rank-b", "rank", status="degraded"), OCT_30, 20, RANKED)
    add_predictions(add_model_run("rank-c", "rank", status="retired"), NOV_2, 20, RANKED)
    add_predictions(add_model_run("rank-d", "rank", status="candidate"), NOV_2, 20, RANKED)
    add_predictions(add_model_run("rank-e", "rank", model_type=SCREENING_MODEL), NOV_2, 20, RANKED)
    add_predictions(add_model_run("rank-f", "rank"), OCT_29, 5, RANKED)
    add_predictions(add_model_run("vol-a", "volatility"), NOV_2, 20, {"2330": {"vol_pred": 0.3}})

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
