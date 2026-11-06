import shutil
from collections.abc import Iterator
from dataclasses import replace
from datetime import date
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.conf import settings as django_settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import transaction
from django.db.models import F
from django.test import override_settings

from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, StockMaster, TradingCalendar
from apps.research import baseline_runs, lgbm_runs, rank_eval
from apps.research.eval_report import render_report
from apps.research.evaluation import RankReport, decile_report, importance_drift, rank_report
from apps.research.features.store import export_year, read_columns, read_partitions
from apps.research.management.commands import evaluate_rank
from apps.research.metrics.ic import ICSummary, daily_rank_ic
from apps.research.models import BacktestMetric, FeatureDaily, LabelDaily, ModelRun
from apps.research.panel import load_price_panel
from apps.research.prediction_cache import MissingFoldsError, fold_path, load_predictions
from apps.research.walkforward import Fold
from tests.research.run_panels import CONFIG, STOCKS, build_panel, folds, research, weekdays

pytestmark = pytest.mark.django_db

INDUSTRIES = ["水泥工業", "食品工業", "塑膠工業", "電子工業"]
RAW_INDUSTRY = {STOCKS[0]: "金融業", STOCKS[1]: "  "}
CANONICAL = {STOCKS[0]: "金融保險", STOCKS[1]: None}


def industry_of(stock: str) -> str | None:
    """The industry evaluation sees: TPEx aliases mapped to TWSE names, blanks missing."""
    return CANONICAL.get(stock, INDUSTRIES[STOCKS.index(stock) % len(INDUSTRIES)])


RUN_KEY = "rank-wf1"


def calendar() -> TradingDays:
    days = weekdays()
    return TradingDays(days, days[0], days[-1])


def add_exposures(directory: Path) -> None:
    """Liquidity, size and beta on every feature row, re-exported to the 2016 partition."""
    rng = np.random.default_rng(11)
    rows = list(FeatureDaily.objects.all())
    for row in rows:
        row.liq_dollar_vol_med_20d = float(rng.lognormal(15, 1))
        row.size_log_mcap = float(rng.normal(22, 1))
        row.risk_beta_250d = float(rng.normal(1, 0.3))
    fields = ["liq_dollar_vol_med_20d", "size_log_mcap", "risk_beta_250d"]
    FeatureDaily.objects.bulk_update(rows, fields, batch_size=2000)
    export_year(2016, directory)


def add_prices() -> None:
    rng = np.random.default_rng(12)
    masters, bars = [], []
    for k, stock in enumerate(STOCKS):
        masters.append(
            StockMaster(
                stock_id=stock,
                name=stock,
                market="TWSE",
                industry=RAW_INDUSTRY.get(stock, INDUSTRIES[k % len(INDUSTRIES)]),
                list_date=date(2000, 1, 1),
            )
        )
        close = 50.0
        for day in weekdays():
            close *= float(np.exp(rng.normal(0, 0.02)))
            opened = close * float(np.exp(rng.normal(0, 0.005)))
            bars.append(
                DailyPrice(
                    stock_id=stock,
                    date=day,
                    open=Decimal(f"{opened:.4f}"),
                    high=Decimal(f"{max(opened, close) * 1.01:.4f}"),
                    low=Decimal(f"{min(opened, close) * 0.99:.4f}"),
                    close=Decimal(f"{close:.4f}"),
                    volume=1000,
                    turnover=int(close * 1000),
                )
            )
    StockMaster.objects.bulk_create(masters)
    DailyPrice.objects.bulk_create(bars, batch_size=5000)


BUILT_TABLES = [DailyPrice, StockMaster, ModelRun, LabelDaily, FeatureDaily, TradingCalendar]


def build_caches(root: Path) -> None:
    """Ridge and LightGBM rank caches of tuning and research folds, with exposures and prices."""
    with override_settings():
        with transaction.atomic():
            panel = build_panel(django_settings, root)
            add_exposures(panel)
            add_prices()
            flipped = 1 + 1 / len(STOCKS) - F("fwd_rank_pct")
            LabelDaily.objects.filter(horizon=5).update(fwd_rank_pct=flipped)
        baseline_runs.run_folds("rank", research(), RUN_KEY, panel)
        lgbm_runs.run_folds("rank", research(), RUN_KEY, panel)


@pytest.fixture(scope="module")
def built(
    tmp_path_factory: pytest.TempPathFactory, django_db_setup: None, django_db_blocker: Any
) -> Iterator[Path]:
    """The caches built and committed once for the module, deleted after its last test."""
    root = tmp_path_factory.mktemp("rank_eval")
    with django_db_blocker.unblock(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(lgbm_runs, "MAX_ROUNDS", 120)
        try:
            build_caches(root)
            yield root
        finally:
            for table in BUILT_TABLES:
                table._default_manager.all().delete()


@pytest.fixture
def caches(built: Path, settings: Any, tmp_path: Path) -> Path:
    """A copy of the built files for each test; database changes roll back with the test."""
    root = tmp_path / "built"
    shutil.copytree(built, root)
    settings.RESEARCH_CACHE_DIR = root / "predictions"
    settings.MODEL_DIR = root / "models"
    settings.FEATURE_PARQUET_DIR = root / "features"
    return root / "features"


def prices(chosen: list[Fold]) -> pd.DataFrame:
    start, end = rank_eval.price_window(chosen, 20, calendar())
    return load_price_panel(calendar(), start, end)


def tradable_ic(frame: pd.DataFrame, score: str, chosen: list[Fold]) -> float:
    """Pooled daily rank IC over research folds' test_start..eval_end, computed apart."""
    inside = np.zeros(len(frame), dtype=bool)
    for fold in chosen:
        if fold.research:
            inside |= frame["date"].between(fold.test_start, fold.eval_end).to_numpy()
    rows = frame[inside & frame["in_universe"].to_numpy() & frame["untradable_t1"].eq(False)]
    return float(daily_rank_ic(rows, score).mean())


def test_read_columns_reads_keys_and_the_named_features_only(caches: Path) -> None:
    start, end = date(2016, 3, 1), date(2016, 3, 31)

    read = read_columns(start, end, caches, ["size_log_mcap", "tech_vol_20d"])

    full = read_partitions(start, end, caches)
    assert list(read.columns) == ["stock_id", "date", "size_log_mcap", "tech_vol_20d"]
    pd.testing.assert_frame_equal(
        read.reset_index(drop=True), full[list(read.columns)].reset_index(drop=True)
    )
    with pytest.raises(ValueError, match="not feature columns"):
        read_columns(start, end, caches, ["fwd_rank_pct"])


@pytest.mark.parametrize("horizon", [5, 20])
def test_evaluation_frame_joins_caches_labels_exposures_and_industry(
    caches: Path, horizon: int
) -> None:
    chosen = research()

    frame = rank_eval.evaluation_frame(RUN_KEY, chosen, horizon, caches)

    cached = load_predictions(RUN_KEY, "baseline", chosen)
    ends = {fold.index: fold.eval_end for fold in chosen}
    kept = cached[cached["date"] <= cached["fold"].map(ends)]
    assert len(frame) == len(kept)
    assert frame["date"].max() <= max(fold.eval_end for fold in chosen)
    keys = ["stock_id", "date"]
    labels = pd.DataFrame(
        list(
            LabelDaily.objects.filter(horizon=horizon).values(
                *keys, "fwd_rank_pct", "untradable_t1"
            )
        )
    )
    exposures = pd.DataFrame(list(FeatureDaily.objects.values(*keys, *rank_eval.EXPOSURES)))
    expected = frame[keys].merge(labels, on=keys, how="left").merge(exposures, on=keys, how="left")
    np.testing.assert_allclose(frame["fwd_rank_pct"], expected["fwd_rank_pct"])
    assert frame["untradable_t1"].tolist() == expected["untradable_t1"].tolist()
    for name in rank_eval.EXPOSURES:
        np.testing.assert_allclose(frame[name], expected[name].astype(float), err_msg=name)
    assert frame["industry"].tolist() == [industry_of(stock) for stock in frame["stock_id"]]
    assert {"金融保險", None} <= set(frame["industry"])
    assert {f"ridge_h{horizon}", f"lightgbm_h{horizon}"} <= set(frame.columns)


def test_evaluate_scores_lightgbm_against_ridge_on_the_frame(caches: Path) -> None:
    chosen = research()
    bars = prices(chosen)

    evaluated = rank_eval.evaluate(RUN_KEY, chosen, 20, caches, bars, calendar())

    frame = rank_eval.evaluation_frame(RUN_KEY, chosen, 20, caches)
    expected = rank_report(frame, "lightgbm_h20", 20, chosen, baseline="ridge_h20")
    assert evaluated.model.total == expected.total
    assert evaluated.model.total.mean == pytest.approx(tradable_ic(frame, "lightgbm_h20", chosen))
    assert evaluated.baseline.total.mean == pytest.approx(tradable_ic(frame, "ridge_h20", chosen))
    assert evaluated.model.compared_folds == len([fold for fold in chosen if fold.research])
    assert evaluated.model.neutralized.n_days > 0
    assert list(evaluated.model.by_liquidity.index) == [1, 2, 3]
    assert list(evaluated.model.by_fold.index) == [fold.index for fold in chosen]
    pd.testing.assert_frame_equal(
        evaluated.deciles, decile_report(frame, "lightgbm_h20", 20, chosen, bars, calendar())
    )


def test_evaluate_reads_each_research_fold_importance_from_its_model_run(caches: Path) -> None:
    chosen = research()

    evaluated = rank_eval.evaluate(RUN_KEY, chosen, 5, caches, prices(chosen), calendar())

    gains = {
        fold.index: ModelRun.objects.get(run_id=lgbm_runs.run_id(RUN_KEY, "rank", 5, fold)).params[
            "importance_gain"
        ]
        for fold in chosen
        if fold.research
    }
    expected = importance_drift(gains)
    pd.testing.assert_frame_equal(evaluated.importance.by_fold, expected.by_fold)
    tuning = next(fold for fold in chosen if fold.tuning)
    assert tuning.index not in evaluated.importance.by_fold.index
    ModelRun.objects.filter(run_id=lgbm_runs.run_id(RUN_KEY, "rank", 5, tuning)).delete()
    rank_eval.evaluate(RUN_KEY, chosen, 5, caches, prices(chosen), calendar())
    ModelRun.objects.filter(run_id=lgbm_runs.run_id(RUN_KEY, "rank", 5, chosen[1])).delete()
    with pytest.raises(LookupError, match=rf"folds \[{chosen[1].index}\]"):
        rank_eval.evaluate(RUN_KEY, chosen, 5, caches, prices(chosen), calendar())


def test_evaluate_run_names_missing_caches_before_anything_else(caches: Path) -> None:
    chosen = research()
    fold_path(RUN_KEY, "lightgbm", chosen[1].index).unlink()

    with pytest.raises(MissingFoldsError, match=rf"rank-wf1/lightgbm .* \[{chosen[1].index}\]"):
        rank_eval.evaluate_run(RUN_KEY, chosen, caches, calendar())


def test_evaluate_run_refuses_caches_fit_on_other_labels(caches: Path) -> None:
    chosen = research()
    fold = chosen[-1]
    LabelDaily.objects.filter(date=fold.test_start, horizon=5).update(fwd_rank_pct=0.5)

    with pytest.raises(ValueError, match=rf"folds \[{fold.index}\] of rank-wf1 .* stale"):
        rank_eval.evaluate_run(RUN_KEY, chosen, caches, calendar())


def test_evaluate_run_refuses_caches_fit_on_other_features(caches: Path) -> None:
    chosen = research()
    FeatureDaily.objects.filter(date=chosen[-1].test_start).update(size_log_mcap=1.0)
    export_year(2016, caches)

    indices = ", ".join(str(fold.index) for fold in chosen)
    with pytest.raises(ValueError, match=rf"folds \[{indices}\] of rank-wf1 .* stale"):
        rank_eval.evaluate_run(RUN_KEY, chosen, caches, calendar())


def test_price_window_runs_from_the_first_research_signal_to_the_last_exit() -> None:
    chosen = research()
    days = calendar()

    start, end = rank_eval.price_window(chosen, 20, days)

    first = next(fold for fold in chosen if fold.research)
    last = max(fold.eval_end for fold in chosen if fold.research)
    assert start == first.test_start
    assert end == days.shift(last, 20)
    assert end < next(fold for fold in folds() if fold.holdout).test_start
    with pytest.raises(ValueError, match="no research folds"):
        rank_eval.price_window([fold for fold in chosen if fold.tuning], 20, days)


def test_write_metrics_store_lightgbm_and_ridge_rows_once(caches: Path) -> None:
    chosen = research()
    evaluations = rank_eval.evaluate_run(RUN_KEY, chosen, caches, calendar())

    written = rank_eval.write_metrics(RUN_KEY, evaluations)
    again = rank_eval.write_metrics(RUN_KEY, evaluations)

    rows = BacktestMetric.objects.filter(run_key=RUN_KEY)
    assert written == again == rows.count()
    assert set(rows.values_list("model_type", "horizon").distinct()) == {
        ("lightgbm", 5),
        ("lightgbm", 20),
        ("ridge", 5),
        ("ridge", 20),
    }
    for horizon, evaluated in evaluations.items():
        for model_type, report in (("lightgbm", evaluated.model), ("ridge", evaluated.baseline)):
            total = rows.get(
                model_type=model_type, horizon=horizon, metric="rank_ic", grain="total", segment=""
            )
            assert total.value == pytest.approx(report.total.mean)
    research_end = max(fold.eval_end for fold in chosen if fold.research)
    assert rows.filter(grain="fold").latest("period_end").period_end <= research_end


def test_report_renders_each_horizon_with_deciles_and_importance(caches: Path) -> None:
    chosen = research()
    evaluations = rank_eval.evaluate_run(RUN_KEY, chosen, caches, calendar())

    text = rank_eval.report(RUN_KEY, evaluations)

    for horizon in (5, 20):
        evaluated = evaluations[horizon]
        section = render_report(
            f"{RUN_KEY}（{horizon} 日）",
            [evaluated.model],
            {horizon: evaluated.deciles},
            evaluated.importance,
        )
        assert section in text
        assert rank_eval._gates(horizon, evaluated) in text
    assert text.index("（5 日）") < text.index("（20 日）")


@pytest.fixture
def command(caches: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(evaluate_rank, "DEFAULT_CONFIG", CONFIG)
    return caches


def test_evaluate_rank_writes_the_report_and_metrics(command: Path, tmp_path: Path) -> None:
    output = tmp_path / "reports" / "rank.md"
    out = StringIO()

    call_command("evaluate_rank", run_key=RUN_KEY, output=str(output), stdout=out)

    evaluations = rank_eval.evaluate_run(RUN_KEY, research(), command, calendar())
    assert output.read_text() == rank_eval.report(RUN_KEY, evaluations)
    assert f"wrote {output}" in out.getvalue()
    assert BacktestMetric.objects.filter(run_key=RUN_KEY, model_type="ridge").exists()


def test_evaluate_rank_writes_nothing_when_the_report_cannot_be_written(
    command: Path, tmp_path: Path
) -> None:
    with pytest.raises(CommandError, match="Is a directory"):
        call_command("evaluate_rank", run_key=RUN_KEY, output=str(tmp_path), stdout=StringIO())
    assert not BacktestMetric.objects.exists()


def test_evaluate_rank_refuses_other_scopes_and_missing_caches(command: Path) -> None:
    with pytest.raises(CommandError, match="start with rank-"):
        call_command("evaluate_rank", run_key="interval-wf1", stdout=StringIO())
    with pytest.raises(CommandError, match=r"rank-none/baseline .* folds"):
        call_command("evaluate_rank", run_key="rank-none", stdout=StringIO())
    with pytest.raises(CommandError, match="no research folds"):
        call_command("evaluate_rank", run_key=RUN_KEY, last_date="2016-09-20", stdout=StringIO())
    assert not BacktestMetric.objects.exists()


def test_evaluate_rank_leaves_holdout_folds_out(command: Path, monkeypatch: Any) -> None:
    seen: list[list[Fold]] = []
    original = rank_eval.evaluate_run

    def recorded(run_key: str, chosen: list[Fold], *args: Any) -> Any:
        seen.append(chosen)
        return original(run_key, chosen, *args)

    monkeypatch.setattr(evaluate_rank, "evaluate_run", recorded)

    call_command("evaluate_rank", run_key=RUN_KEY, stdout=StringIO())

    assert [fold.index for fold in seen[0]] == [fold.index for fold in research()]
    assert not any(fold.holdout for fold in seen[0])


def summary(mean: float, nw_t: float, days: int = 100) -> ICSummary:
    return ICSummary(mean, 0.1, mean / 0.1, nw_t, 0.6, days)


def gated(mean: float, nw_t: float, share: float, compared: int) -> rank_eval.RankEvaluation:
    """A hand-made evaluation holding only what the gates read."""
    empty = pd.DataFrame()
    model = RankReport(
        horizon=20,
        folds=(),
        by_fold=empty,
        by_year=empty,
        total=summary(mean, nw_t),
        positive_share=1.0,
        beat_share=share,
        compared_folds=compared,
        by_liquidity=empty,
        neutralized=summary(0.0, 0.0),
        monthly=pd.Series(dtype=float),
        monthly_days=pd.Series(dtype=int),
    )
    ridge = replace(model, total=summary(0.02, 1.5, 90))
    return rank_eval.RankEvaluation(model, ridge, empty, importance_drift({}))


@pytest.mark.parametrize(
    ("mean", "nw_t", "verdict"),
    [(0.05, 2.0, "未過"), (-0.01, 3.0, "未過"), (0.05, 2.01, "通過"), (float("nan"), 3.0, "未過")],
)
def test_gates_need_a_positive_mean_and_a_newey_west_t_above_two(
    mean: float, nw_t: float, verdict: str
) -> None:
    text = rank_eval._gates(20, gated(mean, nw_t, 0.6, 10))

    assert f"大於 2：{verdict}" in text
    assert "Ridge 研究期 IC 均值 0.0200，NW t 1.50，90 天" in text


@pytest.mark.parametrize(
    ("share", "compared", "shown", "verdict"),
    [
        (0.6, 10, "6／10 摺", "通過"),
        (25 / 42, 42, "25／42 摺", "未過"),
        (float("nan"), 0, "—", "未過"),
    ],
)
def test_gates_need_ridge_beaten_in_sixty_percent_of_compared_folds(
    share: float, compared: int, shown: str, verdict: str
) -> None:
    text = rank_eval._gates(5, gated(0.05, 3.0, share, compared))

    assert f"贏 Ridge 的研究摺 {shown} 達 60%：{verdict}" in text
    assert "## 關卡：排名 5 日" in text
