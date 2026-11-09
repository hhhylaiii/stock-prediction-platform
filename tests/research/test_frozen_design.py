"""The frozen design that training, evaluation and production read (PR 4.8b-1)."""

import math
from dataclasses import replace
from datetime import date, timedelta
from io import StringIO
from itertools import count
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, IndexDaily, StockMaster
from apps.research import baseline_runs, lgbm_runs, target_eval
from apps.research.baseline_runs import BASELINE, run_folds
from apps.research.management.commands import (
    ensemble_check,
    evaluate_rank,
    evaluate_targets,
    train,
    xgb_check,
)
from apps.research.models import ModelRun
from apps.research.prediction_cache import fold_notes
from apps.research.production import DESIGNS, Design, design_for
from apps.research.rank_model import SPEC_RECIPE, Variant
from apps.research.relative_up import RELATIVE_LABEL, relative_up
from apps.research.runs import RunError
from apps.research.target_eval import RELATIVE_NOTE
from apps.research.target_models import TARGET_RECIPE, TargetVariant
from apps.research.training import load_labels
from apps.research.walkforward import WalkForwardConfig, generate_folds
from tests.research.run_panels import CONFIG, research, use_designs, weekdays

OTHER = replace(CONFIG, train_months=5)
RELATIVE = TargetVariant(market=False, relative_up=True)


def test_the_frozen_design_is_what_the_research_period_chose() -> None:
    assert {design.config for design in DESIGNS.values()} == {
        WalkForwardConfig(
            window="rolling",
            train_months=60,
            embargo=20,
            test_months=3,
            step_months=3,
            valid_months=6,
            data_start=date(2010, 1, 1),
            first_test=date(2014, 1, 1),
            tuning_end=date(2015, 12, 31),
            holdout_start=date(2025, 1, 1),
        )
    }
    assert DESIGNS["rank"].variant == Variant(min_rounds=200)
    assert DESIGNS["rank"].target == TARGET_RECIPE
    assert DESIGNS["return"].target == TargetVariant(market=False)
    assert DESIGNS["interval"].target == RELATIVE
    assert {DESIGNS[scope].variant for scope in ("return", "interval")} == {SPEC_RECIPE}
    assert all(design.research_run_key is None for design in DESIGNS.values())


def test_the_spec_design_is_the_recipe_the_choice_was_made_against() -> None:
    assert all(design_for(scope, "spec") == Design() for scope in DESIGNS)
    assert design_for("interval") == DESIGNS["interval"]
    with pytest.raises(ValueError, match="design must be one of"):
        design_for("rank", "w60")


@pytest.fixture
def command(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    StockMaster.objects.create(stock_id="1101", name="1101", market="TWSE", industry="水泥工業")
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    monkeypatch.setattr(train, "shuffle_check", lambda *args: 0.0)
    return run_panel


def _spy(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def record(scope: str, folds: Any, run_key: str, *args: Any, **kwargs: Any) -> list[int]:
        seen.append({"scope": scope, "folds": list(folds), "args": args, "kwargs": kwargs})
        return []

    monkeypatch.setattr(owner, name, record)
    return seen


def _research(config: WalkForwardConfig) -> list[Any]:
    days = weekdays()
    calendar = TradingDays(days, days[0], days[-1])
    return [fold for fold in generate_folds(calendar, days[-1], config) if not fold.holdout]


def test_train_fits_lightgbm_on_the_frozen_design(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_designs(monkeypatch, rank=Design(OTHER, Variant(min_rounds=7)))
    seen = _spy(monkeypatch, lgbm_runs, "run_folds")

    call_command("train", model="lightgbm", scope="rank", run_key="rank-wf1", stdout=StringIO())

    folds = seen[0]["folds"]
    assert [fold.train_start for fold in folds] == [fold.train_start for fold in _research(OTHER)]
    assert seen[0]["args"][-2:] == (Variant(min_rounds=7), TARGET_RECIPE)


def test_train_spec_design_keeps_the_recipe_it_was_chosen_against(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_designs(monkeypatch, rank=Design(OTHER, Variant(min_rounds=7)))
    seen = _spy(monkeypatch, lgbm_runs, "run_folds")

    call_command(
        "train",
        model="lightgbm",
        scope="rank",
        run_key="rank-wf1",
        design="spec",
        stdout=StringIO(),
    )

    folds = seen[0]["folds"]
    assert [fold.train_start for fold in folds] == [fold.train_start for fold in research()]
    assert seen[0]["args"][-2:] == (SPEC_RECIPE, TARGET_RECIPE)


def test_train_fits_baselines_on_the_frozen_folds_and_up_label(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_designs(monkeypatch, interval=Design(OTHER, target=RELATIVE))
    seen = _spy(monkeypatch, train, "run_folds")

    call_command("train", scope="interval", run_key="interval-wf1", stdout=StringIO())

    folds = seen[0]["folds"]
    assert [fold.train_start for fold in folds] == [fold.train_start for fold in _research(OTHER)]
    assert seen[0]["kwargs"]["target"] == RELATIVE


def test_evaluate_rank_reads_the_frozen_folds(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_designs(monkeypatch, rank=Design(OTHER))
    seen: list[list[Any]] = []

    def record(run_key: str, folds: list[Any], *args: Any) -> list[Any]:
        seen.append(folds)
        raise LookupError("stop after the folds")

    monkeypatch.setattr(evaluate_rank, "evaluate_run", record)

    with pytest.raises(CommandError, match="stop after the folds"):
        call_command("evaluate_rank", run_key="rank-wf1")

    assert [fold.train_start for fold in seen[0]] == [fold.train_start for fold in _research(OTHER)]


def test_evaluate_targets_scores_the_frozen_up_label(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_designs(monkeypatch, interval=Design(OTHER, target=RELATIVE))
    seen: list[dict[str, Any]] = []

    def record(run_key: str, folds: list[Any], directory: Path, **kwargs: Any) -> Any:
        seen.append({"folds": folds, **kwargs})
        raise LookupError("stop after the folds")

    monkeypatch.setattr(evaluate_targets, "evaluate_run", record)

    with pytest.raises(CommandError, match="stop after the folds"):
        call_command("evaluate_targets", run_key="interval-wf1")
    with pytest.raises(CommandError, match="stop after the folds"):
        call_command("evaluate_targets", run_key="interval-wf1", design="spec")

    frozen, spec = seen
    assert [f.train_start for f in frozen["folds"]] == [f.train_start for f in _research(OTHER)]
    assert (frozen["relative"], spec["relative"]) == (True, False)


@pytest.fixture
def taiex(run_panel: Path) -> Path:
    """TAIEX_TR closes that sway a few percent, so every logistic CV slice holds both classes."""
    IndexDaily.objects.bulk_create(
        IndexDaily(
            index_id="TAIEX_TR", date=day, close=round(100 * (1 + 0.03 * math.sin(k / 4)), 2)
        )
        for k, day in enumerate(weekdays())
    )
    return run_panel


def test_relative_interval_baselines_fit_the_logistic_on_the_relative_label(
    taiex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[pd.DataFrame] = []
    original = baseline_runs.fit_logistic

    def spy(rows: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        seen.append(rows)
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(baseline_runs, "fit_logistic", spy)
    fold = research()[0]

    run_folds("interval", [fold], "interval-rel", taiex, target=RELATIVE)
    run_folds("interval", [fold], "interval-abs", taiex)

    known = load_labels(fold.train_start, fold.train_end, 20)
    expected = relative_up(known, TradingDays.from_db())
    relative, absolute = seen
    by_row = expected.set_axis(pd.MultiIndex.from_frame(known[["date", "stock_id"]]))
    keys = pd.MultiIndex.from_frame(relative[["date", "stock_id"]])
    assert relative["fwd_up_5pct"].tolist() == by_row.loc[keys].tolist()
    assert (relative["fwd_up_5pct"].to_numpy() != absolute["fwd_up_5pct"].to_numpy()).any()
    logistic = ModelRun.objects.get(run_id=f"interval-rel.logistic.h20.f{fold.index:03d}")
    assert logistic.params["label"] == RELATIVE_LABEL
    rel, abs_ = (fold_notes(key, BASELINE, fold.index) for key in ("interval-rel", "interval-abs"))
    assert rel["recipe"] != abs_["recipe"]
    assert rel["label_hash"].startswith(abs_["label_hash"])
    assert rel["label_hash"] != abs_["label_hash"]
    assert run_folds("interval", [fold], "interval-rel", taiex, target=RELATIVE) == []


def test_baselines_read_only_the_relative_up_flag(taiex: Path) -> None:
    fold = research()[0]

    run_folds("return", [fold], "return-abs", taiex)
    assert (
        run_folds("return", [fold], "return-abs", taiex, target=TargetVariant(market=False)) == []
    )
    with pytest.raises(ValueError, match="relative_up applies to the interval scope"):
        run_folds("return", [fold], "return-rel", taiex, target=RELATIVE)


def test_the_relative_frame_scores_the_up_label_against_taiex(
    taiex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = research()[0]
    lgbm_runs.run_folds("interval", [fold], "interval-rel", taiex, target=RELATIVE)
    run_folds("interval", [fold], "interval-rel", taiex, target=RELATIVE)

    frame = target_eval.target_frame("interval-rel", [fold], taiex, relative=True)
    plain = target_eval.target_frame("interval-rel", [fold], taiex)

    known = load_labels(fold.test_start, fold.eval_end, 20)
    expected = relative_up(known, TradingDays.from_db())
    by_row = expected.set_axis(pd.MultiIndex.from_frame(known[["date", "stock_id"]]))
    keys = pd.MultiIndex.from_frame(frame[["date", "stock_id"]])
    assert frame["fwd_up_5pct"].tolist() == by_row.reindex(keys).tolist()
    assert not frame["fwd_up_5pct"].equals(plain["fwd_up_5pct"])


def test_a_relative_evaluation_refuses_a_run_fit_on_the_absolute_label(taiex: Path) -> None:
    fold = research()[0]
    lgbm_runs.run_folds("interval", [fold], "interval-abs", taiex)
    run_folds("interval", [fold], "interval-abs", taiex)

    lgbm_runs.run_folds("interval", [fold], "interval-rel", taiex, target=RELATIVE)
    run_folds("interval", [fold], "interval-rel", taiex, target=RELATIVE)

    with pytest.raises(ValueError, match="fit on fwd_up_5pct, not fwd_up_5pct_vs_taiex_tr"):
        target_eval.evaluate_run("interval-abs", [fold], taiex, relative=True)
    with pytest.raises(ValueError, match="fit on fwd_up_5pct_vs_taiex_tr, not fwd_up_5pct"):
        target_eval.evaluate_run("interval-rel", [fold], taiex)


def test_train_refuses_a_run_key_cached_for_another_layout(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_designs(monkeypatch, rank=Design(OTHER))
    call_command("train", scope="rank", run_key="rank-wf1", design="spec", stdout=StringIO())

    with pytest.raises(CommandError, match="another walk-forward layout"):
        call_command("train", scope="rank", run_key="rank-wf1", stdout=StringIO())


def test_fits_refuse_another_layout_in_place_but_follow_a_longer_test_period(
    run_panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 60)
    fold = research()[0]
    lgbm_runs.run_folds("rank", [fold], "rank-wf1", run_panel)
    run_folds("rank", [fold], "rank-wf1", run_panel)
    moved = replace(fold, train_start=fold.train_start + timedelta(days=7))

    for runner in (lgbm_runs.run_folds, run_folds):
        with pytest.raises(RunError, match=rf"cached folds \[{fold.index}\] for another"):
            runner("rank", [moved], "rank-wf1", run_panel)
    longer = replace(fold, eval_end=fold.eval_end + timedelta(days=7))
    assert run_folds("rank", [longer], "rank-wf1", run_panel) == [fold.index]
    assert lgbm_runs.run_folds("rank", [moved], "rank-wf1", run_panel, rerun=True) == [fold.index]


def test_the_up_label_check_reads_only_the_folds_it_evaluates(taiex: Path) -> None:
    first, second = research()[:2]
    lgbm_runs.run_folds("interval", [first], "interval-mix", taiex)
    lgbm_runs.run_folds("interval", [second], "interval-mix", taiex, target=RELATIVE)

    target_eval.check_up_label("interval-mix", True, [second])
    with pytest.raises(ValueError, match="fit on fwd_up_5pct, not fwd_up_5pct_vs_taiex_tr"):
        target_eval.check_up_label("interval-mix", True, [first, second])


def test_a_relative_report_names_the_up_label_it_scored(
    command: Path, taiex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 60)
    folds = [fold for fold in research() if fold.research][:1]
    lgbm_runs.run_folds("interval", folds, "interval-rel", taiex, target=RELATIVE)
    run_folds("interval", folds, "interval-rel", taiex, target=RELATIVE)
    evaluation = target_eval.evaluate_run("interval-rel", folds, taiex, relative=True)
    use_designs(monkeypatch, interval=Design(CONFIG, target=RELATIVE))
    monkeypatch.setattr(evaluate_targets, "evaluate_run", lambda *args, **kwargs: evaluation)
    out = StringIO()

    call_command("evaluate_targets", run_key="interval-rel", stdout=out)

    assert RELATIVE_NOTE in out.getvalue()
    assert RELATIVE_NOTE not in target_eval.report("interval-rel", evaluation)


def test_relative_baselines_follow_the_taiex_closes_twenty_sessions_past_train_end(
    taiex: Path,
) -> None:
    fold = research()[0]
    run_folds("interval", [fold], "interval-rel", taiex, target=RELATIVE)
    calendar = TradingDays.from_db()
    beyond = IndexDaily.objects.filter(index_id="TAIEX_TR", date=calendar.shift(fold.train_end, 21))
    reach = IndexDaily.objects.filter(index_id="TAIEX_TR", date=calendar.shift(fold.train_end, 20))

    assert beyond.update(close=2) == 1
    assert run_folds("interval", [fold], "interval-rel", taiex, target=RELATIVE) == []
    assert reach.update(close=2) == 1
    assert run_folds("interval", [fold], "interval-rel", taiex, target=RELATIVE) == [fold.index]


def test_a_taiex_change_while_fitting_a_baseline_names_the_taiex_closes(
    taiex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = count()
    monkeypatch.setattr(baseline_runs, "market_digest", lambda start, end: str(next(calls)))

    with pytest.raises(RuntimeError, match="label_daily or TAIEX_TR changed while fitting fold"):
        run_folds("interval", research()[:1], "interval-rel", taiex, target=RELATIVE)


@pytest.mark.parametrize(("design", "config", "floor"), [("frozen", OTHER, 7), ("spec", CONFIG, 0)])
def test_rank_checks_read_the_folds_of_the_named_design(
    command: Path,
    monkeypatch: pytest.MonkeyPatch,
    design: str,
    config: WalkForwardConfig,
    floor: int,
) -> None:
    use_designs(monkeypatch, rank=Design(OTHER, Variant(min_rounds=7)))
    seen: dict[str, tuple[list[Any], tuple[Any, ...]]] = {}

    def stop(name: str) -> Any:
        def record(run_key: str, folds: list[Any], *args: Any, **kwargs: Any) -> Any:
            seen[name] = (list(folds), args)
            raise ValueError("stop after the folds")

        return record

    monkeypatch.setattr(evaluate_rank, "evaluate_run", stop("evaluate_rank"))
    monkeypatch.setattr(ensemble_check, "ensemble_daily_ics", stop("ensemble_check"))
    monkeypatch.setattr(xgb_check, "xgboost_daily_ics", stop("xgb_check"))

    for name in ("evaluate_rank", "ensemble_check", "xgb_check"):
        with pytest.raises(CommandError, match="stop after the folds"):
            call_command(name, run_key="rank-wf1", design=design, stdout=StringIO())

    def layout(folds: list[Any]) -> set[tuple[int, Any]]:
        return {(fold.index, fold.train_start) for fold in folds}

    chosen = layout(_research(config))
    other = layout(_research(OTHER if config == CONFIG else CONFIG))
    assert set(seen) == {"evaluate_rank", "ensemble_check", "xgb_check"}
    for folds, _ in seen.values():
        assert folds
        assert layout(folds) <= chosen - other
    assert seen["xgb_check"][1][-1] == floor


def test_a_fit_refuses_a_layout_another_model_of_the_key_cached(
    run_panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 60)
    fold = research()[0]
    lgbm_runs.run_folds("rank", [fold], "rank-wf2", run_panel)
    moved = replace(fold, train_start=fold.train_start + timedelta(days=7))

    with pytest.raises(RunError, match="another walk-forward layout"):
        run_folds("rank", [moved], "rank-wf2", run_panel)
