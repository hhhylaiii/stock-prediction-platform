import logging
from datetime import date, datetime
from typing import Any

import pytest
from django.utils import timezone

from apps.etl.models import DataQualityLog
from apps.market.models import TradingCalendar
from apps.research import monitor
from apps.research.models import ModelRun, Prediction
from apps.research.monitor import (
    ALERT_CHECK,
    BRIER_BASELINE,
    DEGRADED_ON,
    DEGRADED_WINDOW,
    IC_THRESHOLD,
    alert_problems,
    run_monitor,
)
from tests.research.test_monitor import (
    DAY,
    alerts,
    calendar,
    calendar_rows,
    dispatched,
    history,
    make_run,
    matured,
    serves,
    status,
)

pytestmark = pytest.mark.django_db
LIMITS = {IC_THRESHOLD: {"20": 0.0}}
FIXTURES = (calendar_rows, dispatched)


def window_days() -> tuple[date, ...]:
    return calendar().between(date(2026, 1, 2), matured(20))[-60:]


def predecessor(rows: int = 60) -> ModelRun:
    """A retired run whose breaching rows fill the oldest part of the slot's window."""
    old = make_run("prod.rank.old", "rank", status="retired", params=LIMITS)
    end = window_days()[rows - 1]
    history(old, "ic", [-0.5] * rows, end=end)
    return old


def judged(entry: dict[str, Any]) -> tuple[Any, ...]:
    return entry["run_id"], entry["mode"], entry["own_rows"], entry["mostly_predecessors"]


def test_the_run_serving_the_day_is_judged_even_without_rows_of_its_own() -> None:
    predecessor()
    new = make_run("prod.rank", "rank", params=LIMITS)
    serves(new, 20)

    result = run_monitor(DAY, calendar())

    assert result.degraded == (new.run_id,)
    [entry] = alerts()
    assert judged(entry) == (new.run_id, "judged", 0, True)
    assert alert_problems(entry) == []
    stored = ModelRun.objects.get(pk=new.pk).params
    assert stored[DEGRADED_ON] == DAY.isoformat()
    assert stored[DEGRADED_WINDOW] == {"own_rows": 0, "days": 60, "mostly_predecessors": True}


def test_a_slot_without_rows_that_day_is_not_judged() -> None:
    predecessor()
    new = make_run("prod.rank", "rank", params=LIMITS)

    result = run_monitor(DAY, calendar())

    assert (result.degraded, alerts(), status(new)) == ((), [], "active")


@pytest.mark.parametrize(("own", "mostly"), [(15, True), (30, False), (45, False)])
def test_the_entry_counts_the_judged_runs_own_rows(own: int, mostly: bool) -> None:
    predecessor(60 - own)
    new = make_run("prod.rank", "rank", params=LIMITS)
    history(new, "ic", [-0.5] * own)

    run_monitor(DAY, calendar())

    [entry] = alerts()
    assert judged(entry) == (new.run_id, "judged", own, mostly)
    assert entry["days"] == 60


@pytest.mark.parametrize(("elapsed", "mode"), [(0, "grace"), (4, "grace"), (5, "judged")])
def test_a_promoted_run_is_spared_for_five_trading_days(
    dispatched: list[DataQualityLog], elapsed: int, mode: str
) -> None:
    predecessor()
    promoted = calendar().shift(DAY, -elapsed)
    new = make_run("prod.rank", "rank", params={**LIMITS, "promoted_on": promoted.isoformat()})
    serves(new, 20)
    serves(new, 20, calendar().shift(DAY, -40))

    result = run_monitor(DAY, calendar())

    [entry] = alerts()
    assert entry["mode"] == mode
    assert alert_problems(entry) == []
    assert [row.check_name for row in dispatched] == [ALERT_CHECK]
    if mode == "grace":
        assert (result.degraded, status(new)) == ((), "active")
        assert entry["grace_start"] == promoted.isoformat()
        assert entry["grace_end"] == calendar().shift(promoted, 4).isoformat()
        assert DEGRADED_ON not in ModelRun.objects.get(pk=new.pk).params
    else:
        assert result.degraded == (new.run_id,)
        assert (entry["grace_start"], entry["grace_end"]) == (None, None)


def test_the_grace_starts_at_the_out_of_sample_boundary_when_it_is_later() -> None:
    predecessor()
    trained = calendar().shift(DAY, -22)
    new = make_run("prod.rank", "rank", params=LIMITS, train_end=trained)
    serves(new, 20)

    run_monitor(DAY, calendar())

    [entry] = alerts()
    assert entry["mode"] == "grace"
    assert entry["grace_start"] == calendar().shift(DAY, -2).isoformat()


@pytest.mark.parametrize(
    ("first", "created", "start"),
    [
        (-2, date(2026, 1, 2), -2),
        (-30, date(2026, 9, 26), date(2026, 9, 28)),
        (-30, date(2026, 9, 29), date(2026, 9, 29)),
    ],
)
def test_a_run_without_promoted_on_starts_its_grace_when_it_first_served(
    first: int, created: date, start: int | date
) -> None:
    predecessor()
    new = make_run("prod.rank", "rank", params=LIMITS)
    ModelRun.objects.filter(pk=new.pk).update(
        params=LIMITS,
        created_at=timezone.make_aware(datetime.combine(created, datetime.min.time())),
    )
    serves(new, 20, calendar().shift(DAY, first))
    serves(new, 20)

    run_monitor(DAY, calendar())

    [entry] = alerts()
    expected = calendar().shift(DAY, start) if isinstance(start, int) else start
    assert (entry["mode"], entry["grace_start"]) == ("grace", expected.isoformat())


@pytest.mark.parametrize(("since", "mode", "reset_rows"), [(11, "reset", 11), (60, "judged", None)])
def test_a_restore_resets_the_window_until_sixty_rows_follow_it(
    since: int, mode: str, reset_rows: int | None
) -> None:
    restored = window_days()[-since]
    run = make_run("prod.rank", "rank", params={**LIMITS, "restored_on": restored.isoformat()})
    history(run, "ic", [-0.5] * 60)

    result = run_monitor(DAY, calendar())

    [entry] = alerts()
    assert alert_problems(entry) == []
    assert (entry["mode"], entry["reset_rows"]) == (mode, reset_rows)
    assert (entry["value"], entry["days"], entry["own_rows"]) == (-0.5, 60, 60)
    if mode == "reset":
        assert (result.degraded, status(run)) == ((), "active")
        assert entry["restored_on"] == restored.isoformat()
        assert entry["reset_needed"] == 60
    else:
        assert result.degraded == (run.run_id,)
        assert (entry["restored_on"], entry["reset_needed"]) == (None, None)


@pytest.mark.parametrize("promoted", [-1, None])
def test_a_successor_keeps_the_reset_window_and_reset_comes_before_grace(
    promoted: int | None,
) -> None:
    days = window_days()
    restored = days[-11]
    old = make_run(
        "prod.rank.old", "rank", status="retired", params={**LIMITS, "restored_on": str(restored)}
    )
    history(old, "ic", [-0.5] * 55, end=days[-6])
    serves(old, 20, restored)
    live = {} if promoted is None else {"promoted_on": calendar().shift(DAY, promoted).isoformat()}
    new = make_run("prod.rank", "rank", params={**LIMITS, **live})
    history(new, "ic", [-0.5] * 5)

    result = run_monitor(DAY, calendar())

    [entry] = alerts()
    assert (entry["run_id"], entry["mode"], entry["reset_rows"]) == (new.run_id, "reset", 11)
    assert (entry["grace_start"], entry["grace_end"]) == (None, None)
    assert result.degraded == ()


def test_the_judged_runs_own_later_restore_keeps_an_earlier_day_from_degrading_it() -> None:
    later = calendar().shift(DAY, 3)
    run = make_run("prod.rank", "rank", params={**LIMITS, "restored_on": later.isoformat()})
    history(run, "ic", [-0.5] * 60)

    result = run_monitor(DAY, calendar())

    [entry] = alerts()
    assert (entry["mode"], entry["restored_on"], entry["reset_rows"]) == (
        "reset",
        later.isoformat(),
        0,
    )
    assert (result.degraded, status(run)) == ((), "active")


def test_another_runs_later_restore_does_not_reset_an_earlier_day() -> None:
    later = calendar().shift(DAY, 3)
    old = make_run(
        "prod.rank.old", "rank", status="retired", params={**LIMITS, "restored_on": str(later)}
    )
    serves(old, 20, window_days()[0])
    run = make_run("prod.rank", "rank", params=LIMITS)
    history(run, "ic", [-0.5] * 60)

    result = run_monitor(DAY, calendar())

    assert [entry["mode"] for entry in alerts()] == ["judged"]
    assert result.degraded == (run.run_id,)


def test_a_restore_of_another_horizon_does_not_reset_the_slot() -> None:
    short = make_run(
        "rank-h5",
        "rank",
        status="retired",
        params={**LIMITS, "restored_on": str(window_days()[-3])},
    )
    serves(short, 5, window_days()[-3])
    run = make_run("rank-h20", "rank", params=LIMITS)
    history(run, "ic", [-0.5] * 60)

    result = run_monitor(DAY, calendar())

    assert [entry["mode"] for entry in alerts()] == ["judged"]
    assert result.degraded == (run.run_id,)


def test_the_latest_of_several_restores_sets_the_reset() -> None:
    days = window_days()
    for name, restored in (("first", days[0]), ("second", days[-11])):
        old = make_run(
            f"prod.rank.{name}",
            "rank",
            status="retired",
            params={**LIMITS, "restored_on": str(restored)},
        )
        serves(old, 20, restored)
    run = make_run("prod.rank", "rank", params=LIMITS)
    history(run, "ic", [-0.5] * 60)

    run_monitor(DAY, calendar())

    [entry] = alerts()
    assert (entry["mode"], entry["restored_on"], entry["reset_rows"]) == (
        "reset",
        str(days[-11]),
        11,
    )


def test_a_reset_alerts_on_the_chained_mean_while_the_restored_days_recover() -> None:
    days = window_days()
    run = make_run("prod.rank", "rank", params={**LIMITS, "restored_on": str(days[-11])})
    history(run, "ic", [-0.5] * 49, end=days[-12])
    history(run, "ic", [0.5] * 11)

    result = run_monitor(DAY, calendar())

    [entry] = alerts()
    assert entry["mode"] == "reset"
    assert entry["value"] == pytest.approx(-19 / 60)
    assert (entry["days"], entry["reset_rows"], result.degraded) == (60, 11, ())


def test_a_day_served_before_the_run_went_live_is_spared() -> None:
    predecessor()
    live = calendar().shift(DAY, 2)
    new = make_run("prod.rank", "rank", params={**LIMITS, "promoted_on": live.isoformat()})
    serves(new, 20)

    result = run_monitor(DAY, calendar())

    [entry] = alerts()
    assert (entry["mode"], entry["grace_start"]) == ("grace", live.isoformat())
    assert entry["grace_end"] == calendar().shift(live, 4).isoformat()
    assert alert_problems(entry) == []
    assert (result.degraded, status(new)) == ((), "active")


def test_without_a_grace_the_serving_run_is_judged_on_its_first_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(monitor, "GRACE_DAYS", 0)
    predecessor()
    new = make_run("prod.rank", "rank", params={**LIMITS, "promoted_on": DAY.isoformat()})
    serves(new, 20)

    result = run_monitor(DAY, calendar())

    assert [entry["mode"] for entry in alerts()] == ["judged"]
    assert result.degraded == (new.run_id,)


def test_an_active_run_without_rows_that_day_is_not_judged() -> None:
    run = make_run("prod.rank", "rank", params=LIMITS)
    history(run, "ic", [-0.5] * 60)
    Prediction.objects.filter(run=run, date=DAY).delete()

    result = run_monitor(DAY, calendar())

    assert (result.degraded, alerts(), status(run)) == ((), [], "active")


def test_a_run_whose_grace_cannot_be_placed_is_skipped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    predecessor()
    new = make_run("prod.rank", "rank", params=LIMITS, train_end=date(2026, 3, 28))
    serves(new, 20)

    with caplog.at_level(logging.WARNING, logger="apps.research.monitor"):
        result = run_monitor(DAY, calendar())

    assert (result.degraded, alerts(), status(new)) == ((), [], "active")
    assert f"run {new.run_id} is not judged on {DAY}" in caplog.text


def test_a_grace_past_the_end_of_the_calendar_warns_and_ends_on_its_last_day(
    caplog: pytest.LogCaptureFixture,
) -> None:
    predecessor()
    last = calendar().shift(DAY, 2)
    TradingCalendar.objects.filter(date__gt=last).delete()
    new = make_run("prod.rank", "rank", params={**LIMITS, "promoted_on": DAY.isoformat()})
    serves(new, 20)

    with caplog.at_level(logging.WARNING, logger="apps.research.monitor"):
        result = run_monitor(DAY, calendar())

    [entry] = alerts()
    assert (entry["mode"], entry["grace_end"]) == ("grace", last.isoformat())
    assert (result.degraded, status(new)) == ((), "active")
    assert f"the calendar ends inside the grace of {new.run_id}" in caplog.text


def test_one_day_writes_judged_grace_and_reset_entries_the_contract_accepts() -> None:
    predecessor()
    rank = make_run("prod.rank", "rank", params=LIMITS)
    serves(rank, 20)
    band = make_run(
        "prod.quantile", "quantile", params={"promoted_on": calendar().shift(DAY, -1).isoformat()}
    )
    history(band, "coverage", [0.5] * 60)
    chance = make_run(
        "prod.classification",
        "classification",
        params={BRIER_BASELINE: 0.2, "restored_on": window_days()[-3].isoformat()},
    )
    history(chance, "brier", [0.3] * 60)

    run_monitor(DAY, calendar())

    entries = alerts()
    assert sorted((entry["objective"], entry["mode"]) for entry in entries) == [
        ("classification", "reset"),
        ("quantile", "grace"),
        ("rank", "judged"),
    ]
    assert [alert_problems(entry) for entry in entries] == [[], [], []]


def entry_of(mode: str, **changed: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "run_id": "prod.rank",
        "objective": "rank",
        "horizon": 20,
        "metric": "ic",
        "reason": "below_threshold",
        "value": -0.5,
        "threshold": 0.0,
        "days": 60,
        "mode": mode,
        "own_rows": 60,
        "mostly_predecessors": False,
        "grace_start": None,
        "grace_end": None,
        "restored_on": None,
        "reset_rows": None,
        "reset_needed": None,
    }
    if mode == "grace":
        entry.update(grace_start="2026-09-28", grace_end="2026-10-02")
    if mode == "reset":
        entry.update(restored_on="2026-09-01", reset_rows=11, reset_needed=60)
    return {**entry, **changed}


@pytest.mark.parametrize(
    ("entry", "problem"),
    [
        ({key: value for key, value in entry_of("judged").items() if key != "days"}, "days"),
        (entry_of("judged", extra=1), "extra"),
        (entry_of("judged", days="60"), "days"),
        (entry_of("judged", own_rows=True), "own_rows"),
        (entry_of("judged", threshold=[0.7, 0.9]), None),
        (entry_of("judged", threshold=None, reason="missing_reference"), None),
        (entry_of("paused"), "mode"),
        (entry_of("grace", grace_end=None), "grace_end"),
        (entry_of("judged", restored_on="2026-09-01"), "restored_on"),
        (entry_of("reset", reset_needed=None), "reset_needed"),
        (entry_of("grace", grace_start="2026/09/28"), "grace_start"),
    ],
)
def test_the_contract_check_names_what_is_wrong(entry: dict[str, Any], problem: str | None) -> None:
    found = alert_problems(entry)

    if problem is None:
        assert found == []
    else:
        assert found
        assert all(problem in item for item in found)


def test_the_contract_lists_the_fields_each_mode_fills() -> None:
    assert set(monitor.MODE_FIELDS) == set(monitor.MODES) == {"judged", "grace", "reset"}
    assert monitor.MODE_FIELDS["judged"] == ()
    filled = {name for names in monitor.MODE_FIELDS.values() for name in names}
    assert filled <= set(monitor.ALERT_FIELDS)
    assert monitor.GRACE_DAYS == 5
