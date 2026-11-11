import math
import re
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.models import IndexDaily
from apps.research import (
    baseline_runs,
    holdout,
    holdout_eval,
    holdout_runs,
    lgbm_runs,
    prediction_cache,
    production,
    rank_eval,
)
from apps.research.eval_report import RANK_METRICS
from apps.research.evaluation import rank_report
from apps.research.holdout_eval import (
    DEGRADED,
    FAILED,
    HANDLING,
    HIGHER,
    INSIDE,
    LOWER,
    PASSED,
    SPEC_LINES,
    Check,
    Flag,
    ScopeVerdict,
    Window,
    flag,
    windows,
)
from apps.research.labels import HORIZONS
from apps.research.management.commands import holdout as holdout_command
from apps.research.models import BacktestMetric, HoldoutRun, MetricGrain
from apps.research.production import Design
from apps.research.target_eval import evaluate_interval, evaluate_return, target_frame
from apps.research.target_models import TargetVariant
from tests.research.run_panels import CONFIG, folds, use_designs, weekdays

DAY = weekdays()[-1].isoformat()
PLAIN = TargetVariant(market=False)
RELATIVE = TargetVariant(market=False, relative_up=True)


def series(values: list[float], first: int = 8) -> pd.Series:
    return pd.Series(values, index=range(first, first + len(values)), dtype=float)


def test_windows_find_the_lowest_and_highest_runs_of_consecutive_folds() -> None:
    values = series([0.1] * 10 + [0.0] * 8 + [0.1] * 10 + [0.3] * 8)

    low, high = windows(values, 8)

    assert (low.first, low.last, high.first, high.last) == (18, 25, 36, 43)
    assert low.mean == pytest.approx(0.0)
    assert high.mean == pytest.approx(0.3)
    assert len(values.rolling(8).mean().dropna()) == 29


def test_windows_refuse_too_few_or_missing_folds() -> None:
    with pytest.raises(ValueError, match="need 8 folds"):
        windows(series([0.1] * 7), 8)
    with pytest.raises(ValueError, match="need 8 folds"):
        windows(series([0.1] * 8 + [math.nan]), 8)


@pytest.mark.parametrize(
    ("better", "holdout", "flagged"),
    [
        (HIGHER, 0.019, True),
        (HIGHER, 0.02, False),
        (LOWER, 0.081, True),
        (LOWER, 0.08, False),
        (INSIDE, 0.019, True),
        (INSIDE, 0.081, True),
        (INSIDE, 0.05, False),
        (HIGHER, math.nan, True),
    ],
)
def test_a_flag_is_raised_only_past_the_bound_on_its_bad_side(
    better: str, holdout: float, flagged: bool
) -> None:
    raised = Flag("m", holdout, better, Window(0.02, 8, 15), Window(0.08, 20, 27))

    assert raised.flagged is flagged


def test_the_flag_weighs_holdout_folds_equally() -> None:
    research = series([0.05] * 36)
    holdout_folds = series([0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02], first=44)

    made = flag("ic", research, holdout_folds, HIGHER)

    assert made.holdout == pytest.approx(0.015)
    assert made.flagged


def test_a_holdout_fold_without_a_value_raises_the_flag() -> None:
    made = flag("ic", series([0.05] * 36), series([0.10, math.nan, 0.10], first=44), HIGHER)

    assert math.isnan(made.holdout)
    assert made.flagged


def verdict(gates: list[bool], flags: list[bool]) -> ScopeVerdict:
    checks = tuple(Check("g", 0.0, "", "", passed) for passed in gates)
    raised = tuple(
        Flag("f", 0.0 if up else 0.05, HIGHER, Window(0.01, 8, 15), Window(0.1, 8, 15))
        for up in flags
    )
    return ScopeVerdict("rank", checks, raised, (), (), pd.DataFrame())


@pytest.mark.parametrize(
    ("gates", "flags", "status"),
    [
        ([True, True], [False, False], PASSED),
        ([True, True], [False, True], DEGRADED),
        ([True, False], [True, True], FAILED),
        ([False], [False], FAILED),
    ],
)
def test_a_failed_gate_outranks_a_flag_and_a_flag_outranks_a_pass(
    gates: list[bool], flags: list[bool], status: str
) -> None:
    assert verdict(gates, flags).status == status


def test_every_cited_spec_line_still_holds_its_clause() -> None:
    lines = (Path(settings.BASE_DIR) / "docs" / "SPEC.md").read_text().splitlines()

    for number, anchor in SPEC_LINES.values():
        assert lines[number - 1].startswith(anchor), (number, lines[number - 1][:40])


@pytest.fixture
def frozen(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """December as holdout; designs frozen on research runs of October and November."""
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 40)
    monkeypatch.setattr(holdout, "SPEC_HOLDOUT_START", CONFIG.holdout_start)
    monkeypatch.setattr(prediction_cache, "SPEC_HOLDOUT_START", CONFIG.holdout_start)
    monkeypatch.setattr(holdout_eval, "WINDOW", 2)
    monkeypatch.setattr(production, "BAGS", 1)
    monkeypatch.setattr(holdout_runs, "latest_price_date", lambda: weekdays()[-1])
    monkeypatch.setattr(holdout_command, "latest_price_date", lambda: weekdays()[-1])
    use_designs(monkeypatch)
    monkeypatch.setattr(
        production,
        "DESIGNS",
        {
            "rank": Design(CONFIG, research_run_key="rank-wf1"),
            "return": Design(CONFIG, target=PLAIN, research_run_key="return-wf1"),
            "interval": Design(CONFIG, target=PLAIN, research_run_key="interval-wf1"),
        },
    )
    return run_panel


def researched(scope: str, directory: Path) -> list[int]:
    design = production.DESIGNS[scope]
    chosen = [fold for fold in folds() if fold.research]
    key = str(design.research_run_key)
    baseline_runs.run_folds(scope, chosen, key, directory, target=design.target)
    lgbm_runs.run_folds(scope, chosen, key, directory, variant=design.variant, target=design.target)
    return [fold.index for fold in chosen]


def run(scope: str, output: Path, **options: object) -> str:
    out = StringIO()
    call_command(
        "holdout",
        scope=scope,
        run_key=f"{scope}-final",
        output=str(output),
        stdout=out,
        **options,
    )
    return out.getvalue()


@pytest.mark.django_db
def test_the_return_holdout_report_applies_the_fixed_criteria(frozen: Path, tmp_path: Path) -> None:
    research = researched("return", frozen)
    report = tmp_path / "return.md"

    printed = run("return", report)

    text = report.read_text()
    held = [fold for fold in folds() if fold.holdout]
    assert f"fitted folds [{held[0].index}], already written []" in printed
    frame = target_frame("return-final", held, frozen, holdout=True)
    pooled = evaluate_return(frame, holdout_eval.scored(held)).mae["zero"]
    assert (
        f"| 5 日 MAE | {pooled.model:.5f} | <= 零預測 {pooled.baseline:.5f} | docs/SPEC.md:337 |"
        in text
    )
    assert f"研究摺 {research[0]}–{research[1]}" in text
    assert text.splitlines()[4] in {f"## 結論：{status}" for status in (PASSED, FAILED, DEGRADED)}
    status = text.splitlines()[4].removeprefix("## 結論：")
    assert HANDLING[status] in text
    assert "docs/SPEC.md:313" in text
    assert "MAE 逐摺差 NW t" in text
    assert "MAE 對過去 60 日平均的逐摺差 NW t" in text
    assert "## 每摺覆蓋率" not in text
    assert "backtest_metric rows" not in printed
    with pytest.raises(CommandError, match="written once"):
        run("return", report)
    elsewhere = tmp_path / "again.md"
    with pytest.raises(CommandError, match=re.escape(f"was reported in {report}")):
        run("return", elsewhere)
    assert not elsewhere.exists()


@pytest.mark.django_db
def test_the_rank_holdout_stores_its_metrics_under_its_own_run_key(
    frozen: Path, tmp_path: Path
) -> None:
    researched("rank", frozen)

    printed = run("rank", tmp_path / "rank.md")

    rows = BacktestMetric.objects.filter(run_key="rank-final", metric__in=RANK_METRICS)
    assert set(rows.values_list("model_type", flat=True)) == {"lightgbm", "ridge"}
    assert set(rows.values_list("horizon", flat=True)) == {5, 20}
    assert f"rank-final: {BacktestMetric.objects.filter(run_key='rank-final').count()}" in printed
    text = (tmp_path / "rank.md").read_text()
    assert "| 排名 5 日 IC |" in text
    assert "| 排名 20 日 IC |" in text
    assert "贏 Ridge 的摺比例" in text
    held = [fold for fold in folds() if fold.holdout]
    research = [fold for fold in folds() if fold.research]
    verdict, _ = holdout_eval.judge(
        "rank", "rank-final", held, "rank-wf1", folds(), frozen, relative=False
    )
    for horizon, gate, raised in zip(HORIZONS, verdict.gates, verdict.flags, strict=True):
        stored = rows.get(
            model_type="lightgbm",
            metric="rank_ic",
            horizon=horizon,
            grain=MetricGrain.TOTAL,
            segment="",
        )
        assert gate.value == pytest.approx(stored.value)
        frame = rank_eval.evaluation_frame("rank-wf1", research, horizon, frozen)
        past = rank_report(frame, f"lightgbm_h{horizon}", horizon, research)
        means = past.by_fold["mean"].loc[[fold.index for fold in research]]
        assert raised.low.mean == pytest.approx(float(means.mean()))


@pytest.mark.django_db
def test_the_interval_holdout_judges_three_gates_and_lists_waived_folds(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id="TAIEX_TR", date=day, close=round(1000 * 0.9977**k, 4))
        for k, day in enumerate(weekdays())
    )
    relative = Design(CONFIG, target=RELATIVE, research_run_key="interval-wf1")
    monkeypatch.setitem(production.DESIGNS, "interval", relative)
    researched("interval", frozen)

    run("interval", tmp_path / "interval.md")

    text = (tmp_path / "interval.md").read_text()
    for name in ("20 日總覆蓋率", "機率 Brier", "波動率 QLIKE"):
        assert f"| {name} |" in text
    assert "## 每摺覆蓋率 70%–90%（docs/SPEC.md:313）：使用者 2026-11-11 豁免一次，不作判定" in text
    held = [fold for fold in folds() if fold.holdout]
    research = [fold for fold in folds() if fold.research]
    assert f"| 摺 {held[0].index} 覆蓋率 |" in text
    assert "| 20 日覆蓋率 |" in text
    assert "Brier 差（模型減 logistic）" in text
    assert "pinball 對歷史分位數" in text
    assert re.search(r"\| 分位數交叉列數 \| \d+ \|", text)
    now = evaluate_interval(
        target_frame("interval-final", held, frozen, True, holdout=True), holdout_eval.scored(held)
    )
    past = evaluate_interval(target_frame("interval-wf1", research, frozen, True), research)
    verdict, _ = holdout_eval.judge(
        "interval", "interval-final", held, "interval-wf1", folds(), frozen, relative=True
    )
    assert verdict.gates[0].value == pytest.approx(now.coverage)
    covered = verdict.flags[-1]
    research_means = past.coverage_by_fold["mean"].loc[[fold.index for fold in research]]
    assert covered.low.mean == pytest.approx(float(research_means.mean()))
    assert covered.holdout == pytest.approx(float(now.coverage_by_fold["mean"].mean()))
    assert [check.value for check in verdict.waived] == pytest.approx(
        now.coverage_by_fold["mean"].tolist()
    )


def rehearse(output: Path | None = None) -> str:
    out = StringIO()
    options = {} if output is None else {"output": str(output)}
    call_command(
        "holdout",
        scope="rank",
        run_key="rank-rehearsal",
        rehearse=True,
        bags=1,
        last_date=DAY,
        stdout=out,
        **options,
    )
    return out.getvalue()


@pytest.mark.django_db
def test_a_rehearsal_reports_without_a_verdict_or_metrics(frozen: Path, tmp_path: Path) -> None:
    researched("rank", frozen)
    fold = max(fold.index for fold in folds() if fold.research)

    printed = rehearse()

    assert f"rank-rehearsal: rehearsed research fold {fold}" in printed
    assert "## 結論：演練不作判定" in printed
    assert "backtest_metric rows" not in printed
    assert not BacktestMetric.objects.filter(run_key="rank-rehearsal").exists()
    assert not HoldoutRun.objects.exists()
    kept = tmp_path / "kept.md"
    kept.write_text("kept")
    with pytest.raises(CommandError, match="written once"):
        rehearse(kept)
    assert kept.read_text() == "kept"


@pytest.mark.django_db
def test_the_holdout_needs_a_report_path_that_does_not_exist(frozen: Path, tmp_path: Path) -> None:
    taken = tmp_path / "taken.md"
    taken.write_text("kept")

    with pytest.raises(CommandError, match="give --output"):
        call_command("holdout", scope="return", run_key="return-final")
    with pytest.raises(CommandError, match="written once"):
        run("return", taken)

    assert taken.read_text() == "kept"
    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
def test_the_holdout_is_refused_before_its_claim_when_a_research_cache_is_gone(
    frozen: Path, tmp_path: Path
) -> None:
    research = researched("return", frozen)
    prediction_cache.fold_path("return-wf1", lgbm_runs.MODEL, research[-1]).unlink()

    with pytest.raises(CommandError, match=rf"folds \[{research[-1]}\]"):
        run("return", tmp_path / "return.md")

    assert not HoldoutRun.objects.exists()
    assert not (tmp_path / "return.md").exists()


@pytest.mark.django_db
def test_a_holdout_that_failed_to_report_resumes_and_reports_once(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    researched("return", frozen)
    report = tmp_path / "return.md"
    judged = holdout_command.judge

    def interrupted(*args: object, **kwargs: object) -> object:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(holdout_command, "judge", interrupted)
    with pytest.raises(CommandError, match="run the same command again to resume"):
        run("return", report)
    assert not report.exists()
    monkeypatch.setattr(holdout_command, "judge", judged)

    printed = run("return", report)

    index = next(fold.index for fold in folds() if fold.holdout)
    assert f"fitted folds [], already written [{index}]" in printed
    assert report.exists()
    assert HoldoutRun.objects.count() == 1


@pytest.mark.django_db
def test_the_return_bound_is_the_worst_research_window_of_fold_mae_gaps(frozen: Path) -> None:
    researched("return", frozen)
    held = [fold for fold in folds() if fold.holdout]
    call_command(
        "holdout",
        scope="return",
        run_key="return-final",
        output=str(frozen.parent / "r.md"),
    )
    research = [fold for fold in folds() if fold.research]

    verdict, reports = holdout_eval.judge(
        "return", "return-final", held, "return-wf1", folds(), frozen, relative=False
    )

    past = evaluate_return(target_frame("return-wf1", research, frozen), research).mae["zero"]
    gaps = (past.by_fold["model"] - past.by_fold["baseline"]).loc[[f.index for f in research]]
    now = evaluate_return(
        target_frame("return-final", held, frozen, holdout=True), holdout_eval.scored(held)
    ).mae["zero"]
    assert reports == {}
    raised = verdict.flags[0]
    assert raised.high.mean == pytest.approx(float(gaps.mean()))
    assert (raised.high.first, raised.high.last) == (research[0].index, research[-1].index)
    assert raised.holdout == pytest.approx(
        float((now.by_fold["model"] - now.by_fold["baseline"]).mean())
    )
    assert raised.flagged is (raised.holdout > raised.high.mean)
    assert verdict.gates[0].value == pytest.approx(now.model)
