import math
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
    lgbm_runs,
    prediction_cache,
    production,
)
from apps.research.eval_report import RANK_METRICS
from apps.research.holdout_eval import (
    DEGRADED,
    FAILED,
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
from apps.research.models import BacktestMetric
from apps.research.production import Design
from apps.research.target_eval import evaluate_return, target_frame
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
        bags=1,
        last_date=DAY,
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
    assert "docs/SPEC.md:313" in text
    assert "MAE 逐摺差 NW t" in text
    assert "## 每摺覆蓋率" not in text
    with pytest.raises(CommandError, match="reported once"):
        run("return", report)


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
    assert "## 每摺覆蓋率 70%–90%：使用者 2026-11-11 豁免一次，不作判定" in text
    held = next(fold for fold in folds() if fold.holdout)
    assert f"| 摺 {held.index} 覆蓋率 |" in text
    assert "| 20 日覆蓋率 |" in text
    assert "Brier 差（模型減 logistic）" in text


@pytest.mark.django_db
def test_a_rehearsal_reports_without_a_verdict_or_metrics(frozen: Path) -> None:
    researched("return", frozen)
    fold = max(fold.index for fold in folds() if fold.research)
    out = StringIO()

    call_command(
        "holdout",
        scope="return",
        run_key="return-rehearsal",
        rehearse=True,
        bags=1,
        last_date=DAY,
        stdout=out,
    )

    printed = out.getvalue()
    assert f"return-rehearsal: rehearsed research fold {fold}" in printed
    assert "## 結論：演練不作判定" in printed
    assert not BacktestMetric.objects.filter(run_key="return-rehearsal").exists()


@pytest.mark.django_db
def test_the_holdout_needs_a_report_path_that_does_not_exist(frozen: Path, tmp_path: Path) -> None:
    taken = tmp_path / "taken.md"
    taken.write_text("kept")

    with pytest.raises(CommandError, match="give --output"):
        call_command("holdout", scope="return", run_key="return-final", last_date=DAY)
    with pytest.raises(CommandError, match="reported once"):
        run("return", taken)

    assert taken.read_text() == "kept"


@pytest.mark.django_db
def test_the_return_bound_is_the_worst_research_window_of_fold_mae_gaps(frozen: Path) -> None:
    researched("return", frozen)
    held = [fold for fold in folds() if fold.holdout]
    call_command(
        "holdout",
        scope="return",
        run_key="return-final",
        bags=1,
        last_date=DAY,
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
