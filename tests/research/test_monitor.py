from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.models import DataQualityLog
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
    ODDS_CHECK,
    WIDTH_CHECK,
    run_monitor,
)

pytestmark = pytest.mark.django_db
DAY = date(2026, 9, 30)
HOLIDAY = date(2026, 9, 25)
STOCKS = [str(1000 + n) for n in range(40)]
OUTSIDE = [str(9000 + n) for n in range(5)]
VOL = 0.3


@pytest.fixture(autouse=True)
def calendar_rows() -> None:
    first = date(2026, 1, 1)
    days = (first + timedelta(days=n) for n in range(365))
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
    run_id: str, objective: str, status: str = "active", params: dict[str, Any] | None = None
) -> ModelRun:
    return ModelRun.objects.create(
        run_id=run_id,
        run_key="prod",
        model_type="lightgbm",
        objective=objective,
        version="2026-09",
        train_start=date(2023, 1, 2),
        train_end=date(2026, 8, 31),
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
    """Labels of every stock; the last universe stock has none and the first is flat."""
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

    kept5 = five[five["in_universe"] & five["fwd_ret_open"].notna()]
    kept20 = twenty[twenty["in_universe"] & twenty["fwd_ret_open"].notna()]
    assert len(kept20) == 39
    rank = stored(runs["rank"], 20)
    assert rank.n_stocks == 39
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
        assert row.n_stocks == 39
    assert (regression.ic, regression.coverage, regression.brier) == (None, None, None)
    assert (rank.mae, classification.mae, volatility.coverage) == (None, None, None)


def test_runs_whose_predictions_have_no_label_yet_are_not_written() -> None:
    rank = make_run("prod.rank", "rank")
    labels = truth(6)
    predict(
        rank, matured(5), 5, outputs(labels, 6)[["stock_id", "in_universe", "score", "rank", "pct"]]
    )
    label(matured(5), 5, labels.assign(fwd_rank_pct=np.nan, fwd_ret_open=np.nan))

    run_monitor(DAY, calendar())

    assert not ModelMonitor.objects.exists()


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


def status(run: ModelRun) -> str:
    return ModelRun.objects.get(pk=run.pk).status


def snapshot() -> tuple[list[Any], list[Any], list[Any]]:
    monitors = list(
        ModelMonitor.objects.order_by("run_id", "date", "horizon").values_list(
            "run_id",
            "date",
            "horizon",
            "ic",
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
    DataQualityLog.objects.create(date=DAY, check_name="row_count", severity="error", detail={})

    run_monitor(DAY, calendar())
    once = snapshot()
    run_monitor(DAY, calendar())

    assert snapshot() == once
    assert DataQualityLog.objects.filter(check_name="row_count", severity="error").count() == 1
    own = DataQualityLog.objects.filter(check_name__in=monitor.CHECKS)
    assert own.exists()
    assert not own.filter(severity="error").exists()


def test_command_runs_a_trading_day_and_refuses_other_days() -> None:
    rank = make_run("prod.rank", "rank")
    seed_matured({"rank": rank}, 5, 9)

    call_command("monitor", date="2026-09-30")

    assert ModelMonitor.objects.filter(run=rank, horizon=5).exists()
    with pytest.raises(CommandError, match="2026-09-25"):
        call_command("monitor", date="2026-09-25")
    with pytest.raises(monitor.MonitorError, match="not a trading day"):
        run_monitor(HOLIDAY, calendar())


@pytest.mark.parametrize("state", ["degraded", "retired"])
def test_every_run_with_predictions_is_scored_whatever_its_status(state: str) -> None:
    run = make_run("prod.rank.old", "rank", status=state)
    seed_matured({"rank": run}, 5, 10)

    run_monitor(DAY, calendar())

    assert ModelMonitor.objects.get(run=run, horizon=5).n_stocks == 39
    assert status(run) == state


def test_runs_without_predictions_that_day_keep_their_monitor_rows() -> None:
    rank = make_run("prod.rank", "rank")
    other = make_run("prod.rank.old", "rank", status="retired")
    seed_matured({"rank": rank}, 5, 11)
    ModelMonitor.objects.create(run=other, date=matured(5), horizon=5, ic=0.02, n_stocks=40)

    run_monitor(DAY, calendar())

    assert ModelMonitor.objects.get(run=other, horizon=5).ic == 0.02
    assert ModelMonitor.objects.filter(run=rank, horizon=5).exists()


def test_consistency_reads_the_active_run_first_and_never_a_retired_one() -> None:
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
    assert not DataQualityLog.objects.filter(check_name=WIDTH_CHECK).exists()
