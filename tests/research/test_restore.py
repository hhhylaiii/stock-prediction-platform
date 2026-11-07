from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.utils import timezone

from apps.api import serving
from apps.research.models import ModelRun, ModelStatus, Prediction
from apps.research.promotion import restore
from apps.research.runs import RunError
from tests.research.run_panels import weekdays

DAYS = weekdays()
SINCE = DAYS[-12]
BACK = DAYS[-4]
SLOT = serving.Slot("quantile", 20)


def degraded(
    run_id: str = "interval-old.lightgbm.quantile.h20.f000",
    status: str = ModelStatus.DEGRADED,
    objective: str = "quantile",
    horizon: int = 20,
    **params: Any,
) -> ModelRun:
    return ModelRun.objects.create(
        run_id=run_id,
        run_key=run_id.split(".", 1)[0],
        model_type="lightgbm",
        objective=objective,
        version="4",
        status=status,
        train_start=DAYS[0],
        train_end=DAYS[100],
        params={"horizon": horizon, "degraded_on": SINCE.isoformat(), **params},
        feature_list=[],
        lib_versions={},
        feature_hash="0" * 64,
    )


def served(run: ModelRun, *days: date) -> None:
    Prediction.objects.bulk_create(
        Prediction(
            run=run, stock_id="1101", date=day, horizon=20, in_universe=True, q10=-0.1, q50=0.0
        )
        for day in days
    )


@pytest.mark.django_db
def test_restore_closes_the_degraded_period_and_reactivates_the_run(run_panel: Path) -> None:
    earlier = {"start": DAYS[-40].isoformat(), "end": DAYS[-30].isoformat(), "reason": "old"}
    window = {"own_rows": 0, "days": 60, "mostly_predecessors": True}
    run = degraded(alert_periods=[earlier], degraded_window=window)
    served(run, DAYS[-8], BACK)

    restored = restore(run.run_id, BACK, "  the window held the previous run's days ")

    assert restored.status == ModelStatus.ACTIVE
    stored = ModelRun.objects.get(pk=run.pk)
    assert stored.status == ModelStatus.ACTIVE
    assert "degraded_on" not in stored.params
    assert "degraded_window" not in stored.params
    assert stored.params["restored_on"] == BACK.isoformat()
    assert stored.params["alert_periods"] == [
        earlier,
        {
            "start": SINCE.isoformat(),
            "end": BACK.isoformat(),
            "reason": "the window held the previous run's days",
        },
    ]
    inside = serving.runs_on(DAYS[-8], [SLOT])[SLOT]
    after = serving.runs_on(BACK, [SLOT])[SLOT]
    assert inside.alert_periods == ((DAYS[-40], DAYS[-30]), (SINCE, BACK))
    assert serving.interval_alert(inside, DAYS[-8]) is True
    assert serving.interval_alert(after, BACK) is False
    assert serving.interval_alert(after, DAYS[-35]) is True


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("active", r"is active; only a degraded run is restored"),
        ("undated", r"has no readable degraded_on"),
        ("early", r"was degraded on .* after"),
        ("taken", r"interval-new\.lightgbm\.quantile\.h20\.f000 is active for quantile 20 日"),
        ("weekend", r"is not a trading day"),
        ("no reason", r"needs a reason"),
        ("unknown", r"no run interval-none"),
        ("periods", r"alert_periods is not a list"),
        ("served later", r"served degraded rows through .*; restore it from then or later"),
        ("future", r"is after today"),
    ],
)
def test_restore_refuses_what_a_person_should_look_at(
    run_panel: Path, case: str, match: str
) -> None:
    run = degraded(status=ModelStatus.ACTIVE if case == "active" else ModelStatus.DEGRADED)
    run_id, day, reason = run.run_id, BACK, "checked by hand"
    if case == "undated":
        ModelRun.objects.filter(pk=run.pk).update(params={"horizon": 20})
    if case == "early":
        day = DAYS[-20]
    if case == "taken":
        degraded("interval-new.lightgbm.quantile.h20.f000", status=ModelStatus.ACTIVE)
    if case == "weekend":
        day = SINCE + timedelta(days=5 - SINCE.weekday())
    if case == "no reason":
        reason = "  "
    if case == "unknown":
        run_id = "interval-none.lightgbm.quantile.h20.f000"
    if case == "served later":
        served(run, BACK)
        day = DAYS[-6]
    if case == "future":
        day = timezone.localdate() + timedelta(days=30)
    if case == "periods":
        ModelRun.objects.filter(pk=run.pk).update(
            params={**run.params, "alert_periods": {"start": "x"}}
        )
    before = ModelRun.objects.get(pk=run.pk)

    with pytest.raises(RunError, match=match):
        restore(run_id, day, reason)

    after = ModelRun.objects.get(pk=run.pk)
    assert (after.status, after.params) == (before.status, before.params)


@pytest.mark.django_db
def test_the_restore_command_reports_the_period_and_its_errors(run_panel: Path) -> None:
    run = degraded()
    out = StringIO()

    call_command(
        "restore_run", run_id=run.run_id, date=BACK.isoformat(), reason="by hand", stdout=out
    )

    assert f"{run.run_id} 自 {BACK} 恢復 active，警示期 {SINCE}–{BACK}" in out.getvalue()
    with pytest.raises(CommandError, match="only a degraded run is restored"):
        call_command("restore_run", run_id=run.run_id, date=BACK.isoformat(), reason="again")


@pytest.mark.django_db
def test_a_run_restored_on_its_degraded_day_keeps_no_alert_period(run_panel: Path) -> None:
    run = degraded()
    served(run, SINCE)

    restore(run.run_id, SINCE, "degraded by mistake")

    stored = ModelRun.objects.get(pk=run.pk)
    assert stored.params["alert_periods"][-1]["start"] == SINCE.isoformat()
    assert stored.params["alert_periods"][-1]["end"] == SINCE.isoformat()
    assert serving.interval_alert(serving.runs_on(SINCE, [SLOT])[SLOT], SINCE) is False


@pytest.mark.django_db
def test_an_active_run_of_another_horizon_does_not_block_a_restore(run_panel: Path) -> None:
    run = degraded("rank-old.lightgbm.rank.h5.f000", objective="rank", horizon=5)
    degraded("rank-new.lightgbm.rank.h20.f000", ModelStatus.ACTIVE, objective="rank", horizon=20)

    restored = restore(run.run_id, BACK, "the h20 run is a different slot")

    assert restored.status == ModelStatus.ACTIVE
