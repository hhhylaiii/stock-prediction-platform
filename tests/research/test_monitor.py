from datetime import date, timedelta
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db.models import F
from django.utils import timezone

from apps.etl.models import DataQualityLog
from apps.etl.quality import CHECKS as ETL_CHECKS
from apps.etl.quality import should_run_prediction
from apps.market.calendar import TradingDays
from apps.market.models import TradingCalendar
from apps.research import monitor
from apps.research.evaluation import band_width
from apps.research.metrics.forecast import (
    daily_absolute_error,
    daily_auc,
    daily_brier,
    daily_coverage,
    daily_direction_hits,
)
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import LabelDaily, ModelMonitor, ModelRun, Prediction
from apps.research.monitor import (
    ALERT_CHECK,
    BRIER_BASELINE,
    DEGRADED_ON,
    IC_THRESHOLD,
    ODDS_CHECK,
    UNSCORED_CHECK,
    WIDTH_CHECK,
    run_monitor,
)
from apps.research.runs import SCREENING_MODEL

pytestmark = pytest.mark.django_db
DAY = date(2026, 9, 30)
HOLIDAY = date(2026, 9, 25)
STOCKS = [str(1000 + n) for n in range(40)]
OUTSIDE = [str(9000 + n) for n in range(5)]
VOL = 0.3


@pytest.fixture(autouse=True)
def calendar_rows() -> None:
    first = date(2025, 1, 1)
    days = (first + timedelta(days=n) for n in range(730))
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5 and day != HOLIDAY)
        for day in days
    )


@pytest.fixture
def dispatched(monkeypatch: pytest.MonkeyPatch) -> list[DataQualityLog]:
    sent: list[DataQualityLog] = []
    monkeypatch.setattr(monitor, "dispatch", lambda rows: sent.extend(rows))
    return sent


def calendar() -> TradingDays:
    return TradingDays.from_db()


def matured(horizon: int) -> date:
    return calendar().shift(DAY, -horizon)


def make_run(
    run_id: str,
    objective: str,
    status: str = "active",
    params: dict[str, Any] | None = None,
    train_end: date = date(2026, 3, 31),
) -> ModelRun:
    return ModelRun.objects.create(
        run_id=run_id,
        run_key="prod",
        model_type="lightgbm",
        objective=objective,
        version="2026-09",
        train_start=date(2023, 1, 2),
        train_end=train_end,
        lib_versions={},
        feature_hash="0" * 64,
        status=status,
        params=params or {},
    )


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    return [{str(key): value for key, value in row.items()} for row in rows]


def predict(run: ModelRun, day: date, horizon: int, frame: pd.DataFrame) -> None:
    Prediction.objects.bulk_create(
        Prediction(run=run, date=day, horizon=horizon, **row) for row in records(frame)
    )


def label(day: date, horizon: int, frame: pd.DataFrame) -> None:
    LabelDaily.objects.bulk_create(
        LabelDaily(date=day, horizon=horizon, **row) for row in records(frame)
    )


def truth(seed: int) -> pd.DataFrame:
    """Labels: the last universe stock has none, the first is flat, the second untradable."""
    rng = np.random.default_rng(seed)
    stocks = [*STOCKS, *OUTSIDE]
    ret = rng.normal(0, 0.1, len(stocks))
    ret[0] = 0.0
    frame = pd.DataFrame(
        {
            "stock_id": stocks,
            "fwd_ret": ret,
            "fwd_ret_open": ret,
            "fwd_rank_pct": pd.Series(ret).rank(pct=True).to_numpy(),
            "fwd_vol_20d": np.abs(rng.normal(VOL, 0.05, len(stocks))),
            "fwd_up_5pct": ret > 0.05,
            "untradable_t1": False,
        }
    )
    missing = frame["stock_id"] == STOCKS[-1]
    frame.loc[missing, ["fwd_ret", "fwd_ret_open", "fwd_rank_pct", "fwd_vol_20d"]] = np.nan
    frame["fwd_up_5pct"] = frame["fwd_up_5pct"].astype(object).where(~missing, None)
    stuck = frame["stock_id"] == STOCKS[1]
    frame.loc[stuck, ["fwd_ret", "fwd_ret_open", "fwd_rank_pct"]] = [0.5, 0.5, np.nan]
    frame.loc[stuck, ["untradable_t1", "fwd_up_5pct"]] = [True, True]
    return frame


def outputs(labels: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Every run's outputs: fair inside the universe, upside down outside it."""
    rng = np.random.default_rng(seed)
    ret = labels["fwd_ret_open"].fillna(0.0).to_numpy()
    sign = np.where(labels["stock_id"].isin(OUTSIDE), -50.0, 1.0)
    score = sign * (ret + rng.normal(0, 0.05, len(ret)))
    q50 = np.clip(sign * ret * 0.3, -0.1, 0.1)
    return pd.DataFrame(
        {
            "stock_id": labels["stock_id"],
            "in_universe": ~labels["stock_id"].isin(OUTSIDE),
            "score": score,
            "rank": pd.Series(-score).rank(method="first").astype(int).to_numpy(),
            "pct": pd.Series(score).rank(pct=True).to_numpy(),
            "ret_pred": sign * ret * 0.5 + rng.normal(0, 0.02, len(ret)),
            "q10": np.where(sign > 0, -0.1, 0.4),
            "q50": q50,
            "q90": np.where(sign > 0, 0.1, 0.4),
            "vol_pred": np.where(sign > 0, VOL, 5.0),
            "prob_up_5pct": np.clip(0.5 + sign * ret, 0.01, 0.99),
        }
    ).assign(q50=lambda frame: frame["q50"].clip(frame["q10"], frame["q90"]))


COLUMNS = {
    "rank": ["score", "rank", "pct"],
    "regression": ["ret_pred"],
    "quantile": ["q10", "q50", "q90"],
    "volatility": ["vol_pred"],
    "classification": ["prob_up_5pct"],
}


def seed_matured(runs: dict[str, ModelRun], horizon: int, seed: int) -> pd.DataFrame:
    day = matured(horizon)
    labels = truth(seed)
    label(day, horizon, labels)
    produced = outputs(labels, seed)
    for objective, run in runs.items():
        predict(run, day, horizon, produced[["stock_id", "in_universe", *COLUMNS[objective]]])
    return produced.merge(labels, on="stock_id").assign(date=day)


def all_runs() -> dict[str, ModelRun]:
    return {objective: make_run(f"prod.{objective}", objective) for objective in COLUMNS}


def stored(run: ModelRun, horizon: int) -> ModelMonitor:
    return ModelMonitor.objects.get(run=run, horizon=horizon)


def first(series: pd.Series) -> float:
    return float(series.iloc[0])


def test_predictions_mature_after_horizon_trading_days() -> None:
    rank = make_run("prod.rank", "rank")
    seed_matured({"rank": rank}, 5, 1)
    seed_matured({"rank": rank}, 20, 2)
    weekday_offset = date(2026, 9, 23)
    labels = truth(3)
    label(weekday_offset, 5, labels)
    predict(
        rank,
        weekday_offset,
        5,
        outputs(labels, 3)[["stock_id", "in_universe", "score", "rank", "pct"]],
    )

    run_monitor(DAY, calendar())

    assert matured(5) == date(2026, 9, 22)
    days = set(ModelMonitor.objects.values_list("horizon", "date"))
    assert days == {(5, matured(5)), (20, matured(20))}


def test_each_objective_is_scored_on_universe_rows_with_a_label() -> None:
    runs = all_runs()
    five = seed_matured({"rank": runs["rank"], "regression": runs["regression"]}, 5, 4)
    twenty = seed_matured({k: v for k, v in runs.items() if k != "regression"}, 20, 5)

    run_monitor(DAY, calendar())

    tradable5, tradable20 = (
        ~five["untradable_t1"].astype(bool),
        ~twenty["untradable_t1"].astype(bool),
    )
    kept5 = five[five["in_universe"] & five["fwd_ret_open"].notna() & tradable5]
    kept20 = twenty[twenty["in_universe"] & twenty["fwd_ret_open"].notna() & tradable20]
    assert len(kept20) == 38
    rank = stored(runs["rank"], 20)
    assert rank.n_stocks == 38
    assert rank.ic == pytest.approx(first(daily_rank_ic(kept20, "score")))
    assert stored(runs["rank"], 5).ic == pytest.approx(first(daily_rank_ic(kept5, "score")))
    regression = stored(runs["regression"], 5)
    assert regression.mae == pytest.approx(
        first(daily_absolute_error(kept5, "fwd_ret_open", "ret_pred"))
    )
    hits = daily_direction_hits(kept5, "fwd_ret_open", "ret_pred")
    assert regression.direction_acc == pytest.approx(first(hits))
    quantile = stored(runs["quantile"], 20)
    assert quantile.coverage == pytest.approx(
        first(daily_coverage(kept20, "fwd_ret_open", "q10", "q90"))
    )
    assert quantile.width == pytest.approx(0.2)
    assert quantile.mae == pytest.approx(first(daily_absolute_error(kept20, "fwd_ret_open", "q50")))
    assert quantile.direction_acc == pytest.approx(
        first(daily_direction_hits(kept20, "fwd_ret_open", "q50"))
    )
    volatility = stored(runs["volatility"], 20)
    assert volatility.mae == pytest.approx(
        first(daily_absolute_error(kept20, "fwd_vol_20d", "vol_pred"))
    )
    chance = kept20.assign(up=kept20["fwd_up_5pct"].astype(float))
    classification = stored(runs["classification"], 20)
    assert classification.brier == pytest.approx(first(daily_brier(chance, "up", "prob_up_5pct")))
    assert classification.auc == pytest.approx(first(daily_auc(chance, "up", "prob_up_5pct")))
    for row in (rank, quantile, volatility, classification):
        assert row.n_stocks == 38
    assert (regression.ic, regression.coverage, regression.brier) == (None, None, None)
    assert (rank.mae, classification.mae, volatility.coverage) == (None, None, None)


def test_predictions_without_labels_are_flagged_and_keep_earlier_scores(
    dispatched: list[DataQualityLog],
) -> None:
    rank = make_run("prod.rank", "rank")
    labels = truth(6)
    predict(
        rank, matured(5), 5, outputs(labels, 6)[["stock_id", "in_universe", "score", "rank", "pct"]]
    )
    label(matured(5), 5, labels.assign(fwd_rank_pct=np.nan, fwd_ret_open=np.nan))
    ModelMonitor.objects.create(run=rank, date=matured(5), horizon=5, ic=0.02, n_stocks=40)

    run_monitor(DAY, calendar())

    assert ModelMonitor.objects.get(run=rank, horizon=5).ic == 0.02
    unscored = logged(UNSCORED_CHECK)
    assert unscored.severity == "warning"
    assert unscored.detail == {
        "runs": [{"run_id": rank.run_id, "date": matured(5).isoformat(), "horizon": 5}]
    }
    assert [row.check_name for row in dispatched] == [UNSCORED_CHECK]


def consistent_outputs(stocks: list[str], in_universe: bool) -> pd.DataFrame:
    half = float(band_width(pd.Series([VOL])).iloc[0]) / 2
    return pd.DataFrame(
        {
            "stock_id": stocks,
            "in_universe": in_universe,
            "q10": -half,
            "q50": 0.0,
            "q90": half,
            "vol_pred": VOL,
            "prob_up_5pct": 0.3,
        }
    )


def seed_today(narrow: int, against: int) -> dict[str, ModelRun]:
    runs = {
        name: make_run(f"prod.{name}", name)
        for name in ("quantile", "volatility", "classification")
    }
    frame = pd.concat(
        [consistent_outputs(STOCKS, True), consistent_outputs(OUTSIDE, False)], ignore_index=True
    )
    frame.loc[: narrow - 1, ["q10", "q90"]] = [-0.001, 0.001]
    frame.loc[narrow : narrow + against - 1, "prob_up_5pct"] = 0.7
    outside = frame["stock_id"].isin(OUTSIDE)
    frame.loc[outside, ["q10", "q90", "prob_up_5pct"]] = [-0.001, 0.001, 0.9]
    for objective, run in runs.items():
        predict(run, DAY, 20, frame[["stock_id", "in_universe", *COLUMNS[objective]]])
    return runs


def logged(check: str) -> DataQualityLog:
    return DataQualityLog.objects.get(date=DAY, check_name=check)


def test_consistency_rows_count_universe_stocks_and_alert_above_ten_percent(
    dispatched: list[DataQualityLog],
) -> None:
    seed_today(narrow=2, against=6)

    run_monitor(DAY, calendar())

    width = logged(WIDTH_CHECK)
    assert width.severity == "warning"
    assert width.detail == {
        "share": 0.05,
        "flagged": 2,
        "n_stocks": 40,
        "stocks": STOCKS[:2],
    }
    odds = logged(ODDS_CHECK)
    assert odds.severity == "warning"
    assert odds.detail["share"] == pytest.approx(0.15)
    assert odds.detail["stocks"] == STOCKS[2:8]
    assert [row.check_name for row in dispatched] == [ODDS_CHECK]


def test_consistency_without_flags_is_logged_as_info_and_without_inputs_not_at_all(
    dispatched: list[DataQualityLog],
) -> None:
    runs = seed_today(narrow=0, against=0)
    Prediction.objects.filter(run=runs["classification"]).delete()

    run_monitor(DAY, calendar())

    width = logged(WIDTH_CHECK)
    assert width.severity == "info"
    assert width.detail == {"share": 0.0, "flagged": 0, "n_stocks": 40, "stocks": []}
    assert not DataQualityLog.objects.filter(check_name=ODDS_CHECK).exists()
    assert dispatched == []


def history(
    run: ModelRun, field: str, values: list[float], horizon: int = 20, end: date | None = None
) -> None:
    last = matured(horizon) if end is None else end
    days = calendar().between(date(2026, 1, 2), last)[-len(values) :]
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run=run, date=day, horizon=horizon, n_stocks=40, **{field: value})
        for day, value in zip(days, values, strict=True)
    )


def status(run: ModelRun) -> str:
    return ModelRun.objects.get(pk=run.pk).status


def alerts_on(day: date) -> list[dict[str, Any]]:
    rows = DataQualityLog.objects.filter(date=day, check_name=ALERT_CHECK)
    return [entry for row in rows for entry in row.detail["runs"]]


def alerts() -> list[dict[str, Any]]:
    rows = DataQualityLog.objects.filter(date=DAY, check_name=ALERT_CHECK)
    assert rows.count() <= 1
    return [entry for row in rows for entry in row.detail["runs"]]


def test_a_rank_run_below_its_rolling_threshold_is_degraded_and_alerted(
    dispatched: list[DataQualityLog],
) -> None:
    run = make_run("prod.rank", "rank", params={IC_THRESHOLD: {"5": -1.0, "20": 0.0}})
    history(run, "ic", [1.0] + [-0.01] * 60)

    run_monitor(DAY, calendar())

    assert status(run) == "degraded"
    assert alerts() == [
        {
            "run_id": run.run_id,
            "objective": "rank",
            "horizon": 20,
            "metric": "ic",
            "value": pytest.approx(-0.01),
            "threshold": 0.0,
            "days": 60,
            "reason": "below_threshold",
        }
    ]
    assert logged(ALERT_CHECK).severity == "warning"
    assert [row.check_name for row in dispatched] == [ALERT_CHECK]


@pytest.mark.parametrize(
    ("values", "params"),
    [
        ([-0.01] * 59, {IC_THRESHOLD: {"20": 0.0}}),
        ([0.01] * 60, {IC_THRESHOLD: {"20": 0.0}}),
        ([0.0] * 60, {IC_THRESHOLD: {"20": 0.0}}),
    ],
)
def test_a_rank_run_short_of_sixty_days_or_at_its_threshold_stays_active(
    values: list[float], params: dict[str, Any]
) -> None:
    run = make_run("prod.rank", "rank", params=params)
    history(run, "ic", values)

    run_monitor(DAY, calendar())

    assert status(run) == "active"
    assert alerts() == []


@pytest.mark.parametrize(
    ("objective", "field", "value", "params", "degraded"),
    [
        ("quantile", "coverage", 0.65, {}, True),
        ("quantile", "coverage", 0.95, {}, True),
        ("quantile", "coverage", 0.70, {}, False),
        ("quantile", "coverage", 0.90, {}, False),
        ("classification", "brier", 0.26, {BRIER_BASELINE: 0.25}, True),
        ("classification", "brier", 0.25, {BRIER_BASELINE: 0.25}, False),
    ],
)
def test_coverage_and_brier_alerts(
    objective: str, field: str, value: float, params: dict[str, Any], degraded: bool
) -> None:
    run = make_run(f"prod.{objective}", objective, params=params)
    history(run, field, [value] * 60)

    run_monitor(DAY, calendar())

    assert status(run) == ("degraded" if degraded else "active")
    assert len(alerts()) == int(degraded)


def test_a_missing_reference_is_a_warning_without_degrading(
    dispatched: list[DataQualityLog],
) -> None:
    rank = make_run("prod.rank", "rank")
    chance = make_run("prod.classification", "classification")
    history(rank, "ic", [-0.5] * 60)
    history(chance, "brier", [0.9] * 60)

    run_monitor(DAY, calendar())

    assert status(rank) == status(chance) == "active"
    reasons = sorted((row["run_id"], row["reason"], row["threshold"]) for row in alerts())
    assert reasons == [
        (chance.run_id, "missing_reference", None),
        (rank.run_id, "missing_reference", None),
    ]
    assert {row.check_name for row in dispatched} == {ALERT_CHECK}


def test_degraded_runs_stay_degraded_and_retired_runs_are_left_alone() -> None:
    limits = {IC_THRESHOLD: {"20": 0.0}}
    kept = make_run("prod.rank.old", "rank", status="degraded", params=limits)
    retired = make_run("prod.rank.retired", "rank", status="retired", params=limits)
    candidate = make_run("prod.rank.next", "rank", status="candidate", params=limits)
    for run in (kept, retired, candidate):
        history(run, "ic", [-0.5] * 60)

    result = run_monitor(DAY, calendar())

    assert [status(run) for run in (kept, retired, candidate)] == [
        "degraded",
        "retired",
        "candidate",
    ]
    assert [row["run_id"] for row in alerts()] == [kept.run_id]
    assert result.degraded == ()


def test_the_window_chains_the_runs_that_served_the_slot() -> None:
    limits = {IC_THRESHOLD: {"20": 0.0}}
    old = make_run("prod.rank.old", "rank", status="retired", params=limits)
    new = make_run("prod.rank", "rank", params=limits)
    history(old, "ic", [-0.5] * 45, end=calendar().shift(matured(20), -15))
    history(new, "ic", [-0.5] * 15)

    result = run_monitor(DAY, calendar())

    assert result.degraded == (new.run_id,)
    assert ModelRun.objects.get(pk=new.pk).params[DEGRADED_ON] == DAY.isoformat()
    assert [(row["run_id"], row["days"]) for row in alerts()] == [(new.run_id, 60)]


def test_in_sample_days_and_days_beyond_six_months_do_not_count() -> None:
    days = calendar().between(date(2026, 1, 2), matured(20))
    limits = {IC_THRESHOLD: {"20": 0.0}}
    trained = make_run("prod.rank", "rank", params=limits, train_end=days[-60])
    history(trained, "ic", [-0.5] * 61)
    run_monitor(DAY, calendar())
    assert (status(trained), alerts()) == ("active", [])

    ModelRun.objects.filter(pk=trained.pk).update(train_end=date(2025, 6, 30))
    ModelMonitor.objects.filter(date__lt=days[-30]).update(date=F("date") - timedelta(days=200))
    run_monitor(DAY, calendar())
    assert (status(trained), alerts()) == ("active", [])


def test_the_active_run_is_judged_before_a_newer_degraded_one_and_screening_runs_never() -> None:
    limits = {IC_THRESHOLD: {"20": 0.0}}
    active = make_run("prod.rank", "rank", params=limits)
    newer = make_run("prod.rank.newer", "rank", status="degraded", params=limits)
    screening = make_run("prod.rank.screen", "rank", params=limits)
    ModelRun.objects.filter(pk=screening.pk).update(model_type=SCREENING_MODEL)
    for run in (active, newer, screening):
        history(run, "ic", [-0.5] * 60)

    result = run_monitor(DAY, calendar())

    assert result.degraded == (active.run_id,)
    assert [row["run_id"] for row in alerts()] == [active.run_id]
    assert status(screening) == "active"


def test_days_after_the_matured_date_do_not_count() -> None:
    run = make_run("prod.rank", "rank", params={IC_THRESHOLD: {"20": 0.0}})
    history(run, "ic", [0.05] * 60)
    later = calendar().between(calendar().shift(matured(20), 1), DAY)
    history(run, "ic", [-0.9] * len(later), end=DAY)

    run_monitor(DAY, calendar())

    assert (status(run), alerts()) == ("active", [])


def test_runs_serving_one_horizon_each_are_judged_on_their_own_slot() -> None:
    limits = {IC_THRESHOLD: {"5": 0.0, "20": 0.0}}
    long = make_run("rank-h20", "rank", params=limits)
    short = make_run("rank-h5", "rank", params=limits)
    history(long, "ic", [-0.5] * 60, horizon=20)
    history(short, "ic", [0.05] * 60, horizon=5)
    out = StringIO()

    call_command("monitor", date=DAY.isoformat(), stdout=out)

    assert (status(long), status(short)) == ("degraded", "active")
    assert [(row["run_id"], row["horizon"]) for row in alerts()] == [(long.run_id, 20)]
    assert "degraded rank-h20" in out.getvalue()


def test_rerunning_a_past_day_judges_the_run_that_served_it_not_todays() -> None:
    limits = {IC_THRESHOLD: {"20": 0.0}}
    old = make_run("prod.rank.old", "rank", "retired", limits, train_end=date(2025, 12, 31))
    new = make_run("prod.rank", "rank", params=limits, train_end=date(2026, 9, 1))
    history(old, "ic", [-0.5] * 60, end=date(2026, 6, 5))
    history(old, "ic", [0.1] * 60)
    past = calendar().shift(date(2026, 6, 5), 20)

    run_monitor(past, calendar())

    assert (status(new), alerts_on(past)) == ("active", [])
    assert DEGRADED_ON not in ModelRun.objects.get(pk=new.pk).params


@pytest.mark.parametrize(("short", "long", "breached"), [(-0.01, -0.01, 5), (0.01, -0.03, 20)])
def test_either_rank_horizon_below_its_own_threshold_degrades(
    short: float, long: float, breached: int
) -> None:
    run = make_run("prod.rank", "rank", params={IC_THRESHOLD: {"5": 0.0, "20": -0.02}})
    history(run, "ic", [short] * 60, horizon=5)
    history(run, "ic", [long] * 60, horizon=20)

    run_monitor(DAY, calendar())

    assert status(run) == "degraded"
    assert [row["horizon"] for row in alerts()] == [breached]


def test_model_checks_never_write_errors_so_the_day_still_predicts() -> None:
    limits = {IC_THRESHOLD: {"5": 0.0, "20": 0.0}, BRIER_BASELINE: 0.1}
    runs = seed_today(narrow=30, against=30)
    ModelRun.objects.update(params=limits)
    history(runs["quantile"], "coverage", [0.2] * 60)
    history(runs["classification"], "brier", [0.9] * 60)
    for run in (make_run("prod.rank", "rank", params=limits), make_run("prod.bare", "rank")):
        history(run, "ic", [-0.5] * 60)
    stale = make_run("prod.regression", "regression")
    predict(stale, matured(5), 5, outputs(truth(13), 13)[["stock_id", "in_universe", "ret_pred"]])
    for name in ETL_CHECKS:
        DataQualityLog.objects.create(date=DAY, check_name=name, severity="info", detail={})

    run_monitor(DAY, calendar())

    written = set(DataQualityLog.objects.filter(date=DAY).values_list("check_name", flat=True))
    assert set(monitor.CHECKS) <= written
    assert not DataQualityLog.objects.filter(severity="error").exists()
    assert should_run_prediction(DAY)


def snapshot() -> tuple[list[Any], list[Any], list[Any]]:
    monitors = list(
        ModelMonitor.objects.order_by("run_id", "date", "horizon").values_list(
            "run_id",
            "date",
            "horizon",
            "ic",
            "direction_acc",
            "coverage",
            "brier",
            "mae",
            "width",
            "auc",
            "n_stocks",
        )
    )
    logs = list(
        DataQualityLog.objects.order_by("check_name", "id").values_list(
            "date", "check_name", "severity", "detail"
        )
    )
    statuses = list(ModelRun.objects.order_by("run_id").values_list("run_id", "status"))
    return monitors, logs, statuses


def test_rerunning_a_day_changes_nothing_and_leaves_other_checks_alone() -> None:
    runs = all_runs()
    seed_matured({"rank": runs["rank"], "regression": runs["regression"]}, 5, 7)
    seed_matured({k: v for k, v in runs.items() if k != "regression"}, 20, 8)
    today = consistent_outputs(STOCKS, True)
    today.loc[:5, "prob_up_5pct"] = 0.8
    for objective in ("quantile", "volatility", "classification"):
        predict(runs[objective], DAY, 20, today[["stock_id", "in_universe", *COLUMNS[objective]]])
    before = calendar().shift(matured(20), -1)
    history(runs["quantile"], "coverage", [0.5] * 59, end=before)
    DataQualityLog.objects.create(date=DAY, check_name="row_count", severity="error", detail={})

    run_monitor(DAY, calendar())
    once = snapshot()
    run_monitor(DAY, calendar())

    assert snapshot() == once
    assert status(runs["quantile"]) == "degraded"
    assert ModelRun.objects.get(pk=runs["quantile"].pk).params[DEGRADED_ON] == DAY.isoformat()
    assert DataQualityLog.objects.filter(check_name="row_count", severity="error").count() == 1
    own = DataQualityLog.objects.filter(check_name__in=monitor.CHECKS)
    assert own.exists()
    assert not own.filter(severity="error").exists()


def test_command_runs_a_trading_day_and_refuses_other_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = make_run("prod.rank", "rank")
    seed_matured({"rank": rank}, 5, 9)

    out = StringIO()
    call_command("monitor", date="2026-09-30", stdout=out)
    assert "degraded none" in out.getvalue()

    assert ModelMonitor.objects.filter(run=rank, horizon=5).exists()
    ModelMonitor.objects.all().delete()
    monkeypatch.setattr(timezone, "localdate", lambda: date(2026, 10, 1))
    call_command("monitor")
    assert ModelMonitor.objects.filter(run=rank, horizon=5, date=matured(5)).exists()
    with pytest.raises(CommandError, match="2026-09-25"):
        call_command("monitor", date="2026-09-25")
    with pytest.raises(monitor.MonitorError, match="not a trading day"):
        run_monitor(HOLIDAY, calendar())


@pytest.mark.parametrize("state", ["degraded", "retired"])
def test_every_run_with_predictions_is_scored_whatever_its_status(state: str) -> None:
    run = make_run("prod.rank.old", "rank", status=state)
    seed_matured({"rank": run}, 5, 10)

    run_monitor(DAY, calendar())

    assert ModelMonitor.objects.get(run=run, horizon=5).n_stocks == 38
    assert status(run) == state


def test_runs_without_predictions_that_day_keep_their_monitor_rows() -> None:
    rank = make_run("prod.rank", "rank")
    other = make_run("prod.rank.old", "rank", status="retired")
    seed_matured({"rank": rank}, 5, 11)
    ModelMonitor.objects.create(run=other, date=matured(5), horizon=5, ic=0.02, n_stocks=40)

    run_monitor(DAY, calendar())

    assert ModelMonitor.objects.get(run=other, horizon=5).ic == 0.02
    assert ModelMonitor.objects.filter(run=rank, horizon=5).exists()


def test_consistency_reads_active_then_degraded_then_retired_runs() -> None:
    runs = seed_today(narrow=0, against=0)
    flagged = consistent_outputs(STOCKS, True).assign(q10=-0.001, q90=0.001, q50=0.0)
    for state in ("degraded", "retired"):
        run = make_run(f"prod.quantile.{state}", "quantile", status=state)
        predict(run, DAY, 20, flagged[["stock_id", "in_universe", "q10", "q50", "q90"]])
    ModelRun.objects.filter(pk=runs["quantile"].pk).update(status="degraded")
    ModelRun.objects.filter(pk="prod.quantile.degraded").update(status="active")

    run_monitor(DAY, calendar())

    assert logged(WIDTH_CHECK).detail["flagged"] == 40
    Prediction.objects.filter(run_id="prod.quantile.degraded").delete()
    run_monitor(DAY, calendar())
    assert logged(WIDTH_CHECK).detail["flagged"] == 0
    Prediction.objects.filter(run=runs["quantile"]).delete()
    run_monitor(DAY, calendar())
    assert logged(WIDTH_CHECK).detail["flagged"] == 40
    ModelRun.objects.filter(pk="prod.quantile.retired").update(status="candidate")
    run_monitor(DAY, calendar())
    assert not DataQualityLog.objects.filter(check_name=WIDTH_CHECK).exists()


def test_rerunning_a_day_after_its_runs_retired_keeps_its_consistency_rows() -> None:
    seed_today(narrow=10, against=0)
    run_monitor(DAY, calendar())
    ModelRun.objects.update(status="retired")

    run_monitor(DAY, calendar())

    assert logged(WIDTH_CHECK).detail["flagged"] == 10


def test_consistency_reads_one_run_per_objective_as_the_api_serves() -> None:
    runs = seed_today(narrow=0, against=0)
    Prediction.objects.filter(run=runs["quantile"], stock_id__in=STOCKS[:10]).delete()
    stale = make_run("prod.quantile.old", "quantile", status="degraded")
    narrow = consistent_outputs(STOCKS[:10], True).assign(q10=-0.001, q90=0.001)
    predict(stale, DAY, 20, narrow[["stock_id", "in_universe", "q10", "q50", "q90"]])

    run_monitor(DAY, calendar())

    assert logged(WIDTH_CHECK).detail["flagged"] == 0
    assert logged(WIDTH_CHECK).detail["n_stocks"] == 30
    newer = make_run("prod.quantile.new", "quantile")
    ModelRun.objects.filter(pk=newer.pk).update(
        created_at=runs["quantile"].created_at + timedelta(days=1)
    )
    predict(
        newer,
        DAY,
        20,
        consistent_outputs(STOCKS, True).assign(q10=-0.001, q90=0.001)[
            ["stock_id", "in_universe", "q10", "q50", "q90"]
        ],
    )
    run_monitor(DAY, calendar())
    assert logged(WIDTH_CHECK).detail["flagged"] == 40


def test_a_run_with_only_out_of_universe_rows_is_reported_unscored(
    dispatched: list[DataQualityLog],
) -> None:
    rank = make_run("prod.rank", "rank")
    labels = truth(12)
    produced = outputs(labels, 12).assign(in_universe=False)
    predict(rank, matured(5), 5, produced[["stock_id", "in_universe", "score", "rank", "pct"]])
    label(matured(5), 5, labels)

    run_monitor(DAY, calendar())

    assert not ModelMonitor.objects.exists()
    assert logged(UNSCORED_CHECK).detail["runs"] == [
        {"run_id": rank.run_id, "date": matured(5).isoformat(), "horizon": 5}
    ]
