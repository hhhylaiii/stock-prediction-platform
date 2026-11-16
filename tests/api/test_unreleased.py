from datetime import date
from typing import Any

import pytest
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.api.unreleased import unreleased
from apps.research.models import HoldoutRelease, HoldoutRun, ModelRun, Prediction
from tests.api.conftest import Runs, envelope_of

pytestmark = pytest.mark.usefixtures("calendar_in_db")

INTERVAL_KEY = "interval-holdout-20261112"
INTERVAL_SLOTS = [
    {"objective": "quantile", "horizon": 20},
    {"objective": "volatility", "horizon": 20},
    {"objective": "classification", "horizon": 20},
]
HELD = [{"scope": "interval", "run_key": INTERVAL_KEY, "slots": INTERVAL_SLOTS}]
PATHS = ("/api/models", "/api/predictions?horizon=20", "/api/stocks/2330/predictions")


def claim(scope: str = "interval", attempt: int = 1, run_key: str = INTERVAL_KEY) -> HoldoutRun:
    reason = "" if attempt == 1 else "a rerun after a broken fold"
    return HoldoutRun.objects.create(
        scope=scope, attempt=attempt, run_key=run_key, last_date=date(2026, 11, 11), reason=reason
    )


def release(holdout: HoldoutRun) -> HoldoutRelease:
    return HoldoutRelease.objects.create(
        holdout=holdout, status="明顯退化", report_sha256="ab" * 32, reason="kept"
    )


def take_interval_offline(runs: Runs) -> None:
    """The interval scope never built: none of its three runs serve."""
    ModelRun.objects.filter(pk__in=[runs.quantile.pk, runs.vol.pk, runs.prob.pk]).update(
        status="candidate"
    )


def everywhere(value: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    return [(path, value) for path in PATHS]


def listed(client: Client) -> list[tuple[str, list[dict[str, Any]]]]:
    return [(path, envelope_of(client.get(path))["meta"]["unreleased"]) for path in PATHS]


def test_a_judged_scope_never_released_with_nothing_serving_is_listed(
    client: Client, runs: Runs
) -> None:
    take_interval_offline(runs)
    claim()
    claim("rank", run_key="rank-holdout-20261112")

    assert listed(client) == everywhere(HELD)


def test_a_released_scope_is_not_listed(client: Client, runs: Runs) -> None:
    take_interval_offline(runs)
    release(claim())

    assert listed(client) == everywhere([])


@pytest.mark.parametrize("serving", ["quantile", "vol", "prob"])
def test_one_serving_run_of_the_scope_takes_it_off_the_list(
    client: Client, runs: Runs, serving: str
) -> None:
    take_interval_offline(runs)
    ModelRun.objects.filter(pk=getattr(runs, serving).pk).update(status="degraded")
    claim()

    assert listed(client) == everywhere([])


def test_a_serving_run_without_prediction_rows_does_not_count(client: Client, runs: Runs) -> None:
    take_interval_offline(runs)
    Prediction.objects.filter(run=runs.vol).delete()
    ModelRun.objects.filter(pk=runs.vol.pk).update(status="active")
    claim()

    assert listed(client) == everywhere(HELD)


def test_no_claim_lists_nothing(client: Client, runs: Runs) -> None:
    take_interval_offline(runs)

    assert listed(client) == everywhere([])


def test_the_latest_attempt_is_the_one_judged(client: Client, runs: Runs) -> None:
    take_interval_offline(runs)
    release(claim())
    claim(attempt=2, run_key="interval-holdout-20261201")

    assert unreleased() == [
        {"scope": "interval", "run_key": "interval-holdout-20261201", "slots": INTERVAL_SLOTS}
    ]


def test_the_five_day_table_carries_the_list_too(client: Client, runs: Runs) -> None:
    take_interval_offline(runs)
    claim()

    assert envelope_of(client.get("/api/predictions?horizon=5"))["meta"]["unreleased"] == HELD


def test_without_a_claim_the_list_costs_one_query(
    runs: Runs, django_assert_num_queries: DjangoAssertNumQueries
) -> None:
    with django_assert_num_queries(1):
        unreleased()


def test_with_a_held_claim_the_list_costs_two_queries(
    runs: Runs, django_assert_num_queries: DjangoAssertNumQueries
) -> None:
    take_interval_offline(runs)
    claim()

    with django_assert_num_queries(2):
        assert unreleased() == HELD
