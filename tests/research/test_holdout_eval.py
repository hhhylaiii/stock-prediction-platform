import hashlib
import json
import math
import re
from dataclasses import replace
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
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
    target_eval,
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
    SPEC_CLAUSES,
    Check,
    Flag,
    Header,
    ScopeVerdict,
    Window,
    flag,
    windows,
)
from apps.research.labels import HORIZONS
from apps.research.management.commands import holdout as holdout_command
from apps.research.models import BacktestMetric, HoldoutRun, LabelDaily, MetricGrain
from apps.research.production import Design
from apps.research.target_eval import evaluate_interval, evaluate_return, target_frame
from apps.research.target_models import TargetVariant
from apps.research.walkforward import Fold, generate_folds
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

    made = flag("ic", windows(research, 8), holdout_folds, HIGHER)

    assert made.holdout == pytest.approx(0.015)
    assert made.flagged


def test_a_holdout_fold_without_a_value_raises_the_flag() -> None:
    bounds = windows(series([0.05] * 36), 8)

    made = flag("ic", bounds, series([0.10, math.nan, 0.10], first=44), HIGHER)

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


def test_the_marked_report_lists_the_gates_that_failed_and_the_flags_raised() -> None:
    passed = {"a": True, "b": False, "c": False}
    gates = tuple(Check(name, 0.0, "", "", met) for name, met in passed.items())
    bound = Window(0.01, 8, 15)
    flags = (Flag("x", 0.0, HIGHER, bound, bound), Flag("y", 0.05, HIGHER, bound, bound))
    judged = ScopeVerdict("rank", gates, flags, (), (), pd.DataFrame())

    marked = holdout_eval.marked_report(judged, "out.md", "the text")

    assert (marked.scope, marked.status) == ("rank", FAILED)
    assert (marked.failed, marked.flagged) == (["b", "c"], ["x"])
    assert marked.sha256 == hashlib.sha256(b"the text").hexdigest()
    assert (marked.text, Path(marked.path).is_absolute()) == ("the text", True)


def test_a_short_verdict_is_reported_as_refused_until_the_user_releases_it() -> None:
    for status in (FAILED, DEGRADED):
        assert "預設不上線" in HANDLING[status]
        assert "`release_holdout`" in HANDLING[status]
    assert "release_holdout" not in HANDLING[PASSED]
    assert "不上線" not in HANDLING[PASSED]


def test_every_cited_spec_clause_opens_one_line_of_its_section() -> None:
    lines = (Path(settings.BASE_DIR) / "docs" / "SPEC.md").read_text().splitlines()

    for section, opening in SPEC_CLAUSES.values():
        found = [index for index, line in enumerate(lines) if line.startswith(opening)]
        assert len(found) == 1, (opening, found)
        headings = [line for line in lines[: found[0]] if line.startswith("## ")]
        assert headings[-1].startswith(f"## {section}. "), (opening, headings[-1])


def test_a_citation_names_the_section_and_clause_without_a_line_number() -> None:
    assert holdout_eval.cite("record") == "docs/SPEC.md 第 7 節「研究期與最終 holdout」"
    assert holdout_eval.cite("rank") == "docs/SPEC.md 第 7 節「排名」"
    assert holdout_eval.cite("return") == "docs/SPEC.md 第 7 節「絕對報酬（5 日）」"
    assert holdout_eval.cite("bagging") == "docs/SPEC.md 第 7 節「LightGBM seed bagging」"
    assert holdout_eval.cite("acceptance") == "docs/SPEC.md 第 9 節「Phase 4 主模型」"
    for name in SPEC_CLAUSES:
        assert ":" not in holdout_eval.cite(name), name


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


def marker(run_key: str) -> dict[str, str]:
    path = Path(settings.RESEARCH_CACHE_DIR) / run_key / "holdout.report"
    stored: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return stored


def held_folds() -> list[Fold]:
    return [fold for fold in folds() if fold.holdout]


def settled(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rows whose label had settled by the panel's last day, as the holdout counts them."""
    return frame[frame["date"] <= TradingDays.from_db().shift(weekdays()[-1], -horizon)]


def judged(scope: str, panel: Path, relative: bool = False) -> ScopeVerdict:
    key = f"{scope}-wf1"
    bounds = holdout_eval.research_bounds(scope, key, folds(), panel, relative)
    header = Header(scope, f"{scope}-final", key, 1, weekdays()[-1], rehearsal=False)
    calendar = TradingDays.from_db()
    return holdout_eval.judge(header, held_folds(), bounds, panel, relative, calendar)[0]


def gap(compared: target_eval.Comparison) -> pd.Series:
    return compared.by_fold["model"] - compared.by_fold["baseline"]


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
    held = held_folds()
    assert f"fitted folds [{held[0].index}], already written []" in printed
    loaded = target_frame("return-final", held, frozen, holdout=True)
    pooled = evaluate_return(settled(loaded, 5), holdout_eval.scored(held)).mae["zero"]
    later = evaluate_return(loaded, holdout_eval.scored(held)).mae["zero"]
    assert pooled.model != later.model
    assert (
        f"| 5 日 MAE | {pooled.model:.5f} | <= 零預測 {pooled.baseline:.5f} | "
        "docs/SPEC.md 第 7 節「每月重訓」 |" in text
    )
    assert f"研究摺 {research[0]}–{research[1]}" in text
    assert text.splitlines()[4] in {f"## 結論：{status}" for status in (PASSED, FAILED, DEGRADED)}
    status = text.splitlines()[4].removeprefix("## 結論：")
    assert HANDLING[status] in text
    assert "docs/SPEC.md 第 7 節「研究期與最終 holdout」" in text
    assert "docs/SPEC.md:" not in text
    assert "MAE 逐摺差 NW t" in text
    assert "MAE 對過去 60 日平均的逐摺差 NW t" in text
    assert "## 每摺覆蓋率" not in text
    assert "backtest_metric rows" not in printed
    days = settled(loaded, 5)["date"].nunique()
    assert f"天數最少的摺：h5 摺 {held[0].index}（{days} 天）。" in text
    assert "逐摺等權平均不依天數加權" in text
    assert "只計 claim 的 last_date 當時 label 已到期的日子" in text
    verdict = judged("return", frozen)
    assert status == verdict.status
    assert marker("return-final") == {
        "scope": "return",
        "status": status,
        "path": str(report),
        "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "text": text,
        "failed": [gate.name for gate in verdict.gates if not gate.passed],
        "flagged": [raised.name for raised in verdict.flags if raised.flagged],
    }
    assert Path(marker("return-final")["path"]).is_absolute()
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
    held = held_folds()
    research = [fold for fold in folds() if fold.research]
    verdict = judged("rank", frozen)
    for horizon, gate, raised in zip(HORIZONS, verdict.gates, verdict.flags, strict=True):
        stored = rows.get(
            model_type="lightgbm",
            metric="rank_ic",
            horizon=horizon,
            grain=MetricGrain.TOTAL,
            segment="",
        )
        loaded = rank_eval.evaluation_frame("rank-final", held, horizon, frozen)
        column = f"lightgbm_h{horizon}"
        now = rank_report(settled(loaded, horizon), column, horizon, holdout_eval.scored(held))
        assert gate.value == pytest.approx(now.total.mean)
        assert gate.value == pytest.approx(stored.value)
        assert gate.passed is bool(now.total.mean > 0)
        frame = rank_eval.evaluation_frame("rank-wf1", research, horizon, frozen)
        past = rank_report(frame, column, horizon, research)
        means = past.by_fold["mean"].loc[[fold.index for fold in research]]
        assert raised.better == HIGHER
        assert raised.low.mean == pytest.approx(float(means.mean()))
        assert raised.holdout == pytest.approx(float(now.by_fold["mean"].mean()))
        assert raised.flagged is bool(raised.holdout < raised.low.mean)


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
    waived = "## 每摺覆蓋率 70%–90%（docs/SPEC.md 第 7 節「研究期與最終 holdout」）"
    assert f"{waived}：使用者 2026-11-11 豁免一次，不作判定" in text
    held = held_folds()
    research = [fold for fold in folds() if fold.research]
    assert f"| 摺 {held[0].index} 覆蓋率 |" in text
    assert "| 20 日覆蓋率 |" in text
    assert "Brier 差（模型減 logistic）" in text
    assert "pinball 對歷史分位數" in text
    assert re.search(
        r"\| 分位數交叉列數 \| \d+ \| = 0 \| docs/SPEC.md 第 9 節「Phase 4 主模型」 \|", text
    )
    loaded = target_frame("interval-final", held, frozen, True, holdout=True)
    now = evaluate_interval(settled(loaded, 20), holdout_eval.scored(held))
    past = evaluate_interval(target_frame("interval-wf1", research, frozen, True), research)
    verdict = judged("interval", frozen, relative=True)
    coverage, brier, qlike = verdict.gates
    assert coverage.value == pytest.approx(now.coverage)
    assert coverage.passed is bool(0.70 <= now.coverage <= 0.90)
    assert (brier.value, qlike.value) == pytest.approx((now.brier.model, now.qlike.model))
    assert brier.passed is bool(now.brier.model <= now.brier.baseline)
    assert qlike.passed is bool(now.qlike.model <= now.qlike.baseline)
    indexes = [fold.index for fold in research]
    assert [raised.better for raised in verdict.flags] == [LOWER, LOWER, INSIDE]
    for raised, before, after in zip(
        verdict.flags[:2], (past.pinball, past.brier), (now.pinball, now.brier), strict=True
    ):
        assert raised.high.mean == pytest.approx(float(gap(before).loc[indexes].mean()))
        assert raised.holdout == pytest.approx(float(gap(after).mean()))
        assert raised.flagged is bool(raised.holdout > raised.high.mean)
    covered = verdict.flags[-1]
    research_means = past.coverage_by_fold["mean"].loc[indexes]
    assert covered.low.mean == pytest.approx(float(research_means.mean()))
    assert covered.holdout == pytest.approx(float(now.coverage_by_fold["mean"].mean()))
    assert covered.flagged is bool(
        covered.holdout < covered.low.mean or covered.holdout > covered.high.mean
    )
    assert [check.value for check in verdict.waived] == pytest.approx(
        now.coverage_by_fold["mean"].tolist()
    )
    assert [check.passed for check in verdict.waived] == [
        bool(0.70 <= value <= 0.90) for value in now.coverage_by_fold["mean"]
    ]


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


def stopped(*args: object, **kwargs: object) -> None:
    raise RuntimeError("interrupted")


@pytest.mark.django_db
def test_a_report_cut_short_after_its_marker_is_written_from_the_marker_alone(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    researched("rank", frozen)
    report, other = tmp_path / "rank.md", tmp_path / "other.md"
    monkeypatch.setattr(holdout_command, "write_once", stopped)
    with pytest.raises(CommandError, match="run the same command again to resume"):
        run("rank", report)
    marked = marker("rank-final")
    assert not report.exists()
    assert BacktestMetric.objects.filter(run_key="rank-final").exists()
    monkeypatch.setattr(holdout_command, "write_once", holdout_eval.write_once)
    monkeypatch.setattr(holdout_command, "judge", stopped)
    monkeypatch.setattr(holdout_command, "run_holdout", stopped)

    with pytest.raises(CommandError, match=re.escape(f"is marked for {report}")):
        run("rank", other)
    run("rank", report)

    assert not other.exists()
    assert report.read_text() == marked["text"]
    assert marker("rank-final") == marked
    assert HoldoutRun.objects.count() == 1
    with pytest.raises(CommandError, match=re.escape(f"was reported in {report}")):
        run("rank", other)


@pytest.mark.django_db
def test_a_marker_that_failed_leaves_no_report_and_the_command_reports_again(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    researched("return", frozen)
    report = tmp_path / "return.md"
    monkeypatch.chdir(tmp_path)

    def killed(*args: object, **kwargs: object) -> None:
        raise OSError("killed before the marker")

    monkeypatch.setattr(holdout_command, "mark_reported", killed)
    with pytest.raises(CommandError, match="killed before the marker"):
        run("return", Path("return.md"))
    assert not report.exists()
    assert holdout_eval.reported("return-final") is None
    monkeypatch.setattr(holdout_command, "mark_reported", holdout_eval.mark_reported)

    run("return", Path("return.md"))

    assert marker("return-final")["text"] == report.read_text()
    assert marker("return-final")["path"] == str(report.resolve())
    assert HoldoutRun.objects.count() == 1


@pytest.mark.django_db
def test_a_research_run_the_bounds_cannot_be_read_from_stops_the_holdout_before_its_claim(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id="TAIEX_TR", date=day, close=round(1000 * 0.9977**k, 4))
        for k, day in enumerate(weekdays())
    )
    researched("interval", frozen)
    researched("return", frozen)
    relative = Design(CONFIG, target=RELATIVE, research_run_key="interval-wf1")
    monkeypatch.setitem(production.DESIGNS, "interval", relative)
    first = next(fold for fold in folds() if fold.research)

    with pytest.raises(CommandError, match="probability models were fit on fwd_up_5pct, not"):
        run("interval", tmp_path / "interval.md")
    LabelDaily.objects.filter(date=first.train_end, horizon=5).update(fwd_ret_open=0.5)
    with pytest.raises(CommandError, match=rf"folds \[{first.index}.*are stale"):
        run("return", tmp_path / "return.md")

    assert not HoldoutRun.objects.exists()
    assert not list(tmp_path.glob("*.md"))


@pytest.mark.django_db
def test_a_rehearsal_on_an_earlier_last_date_reads_the_bounds_of_today_s_folds(
    frozen: Path,
) -> None:
    researched("rank", frozen)
    earlier = next(fold for fold in folds() if fold.research).test_end
    out = StringIO()

    call_command(
        "holdout",
        scope="rank",
        run_key="rank-rehearsal",
        rehearse=True,
        bags=1,
        last_date=earlier.isoformat(),
        stdout=out,
    )

    assert "## 結論：演練不作判定" in out.getvalue()


@pytest.mark.django_db
def test_the_return_bound_is_the_worst_research_window_of_fold_mae_gaps(frozen: Path) -> None:
    researched("return", frozen)
    held = held_folds()
    call_command(
        "holdout",
        scope="return",
        run_key="return-final",
        output=str(frozen.parent / "r.md"),
    )
    research = [fold for fold in folds() if fold.research]

    verdict = judged("return", frozen)

    past = evaluate_return(target_frame("return-wf1", research, frozen), research).mae["zero"]
    gaps = gap(past).loc[[f.index for f in research]]
    loaded = target_frame("return-final", held, frozen, holdout=True)
    now = evaluate_return(settled(loaded, 5), holdout_eval.scored(held)).mae["zero"]
    raised = verdict.flags[0]
    assert raised.better == LOWER
    assert raised.high.mean == pytest.approx(float(gaps.mean()))
    assert (raised.high.first, raised.high.last) == (research[0].index, research[-1].index)
    assert raised.holdout == pytest.approx(float(gap(now).mean()))
    assert raised.flagged is bool(raised.holdout > raised.high.mean)
    assert verdict.gates[0].value == pytest.approx(now.model)
    assert verdict.gates[0].passed is bool(now.model <= now.baseline)


WIDE = replace(CONFIG, tuning_end=date(2016, 8, 31), holdout_start=date(2016, 11, 1))


@pytest.fixture
def wide(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """November and December as holdout, two research folds, each its own window."""
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 40)
    monkeypatch.setattr(holdout, "SPEC_HOLDOUT_START", WIDE.holdout_start)
    monkeypatch.setattr(prediction_cache, "SPEC_HOLDOUT_START", WIDE.holdout_start)
    monkeypatch.setattr(holdout_eval, "WINDOW", 1)
    monkeypatch.setattr(production, "BAGS", 1)
    monkeypatch.setattr(holdout_runs, "latest_price_date", lambda: weekdays()[-1])
    monkeypatch.setattr(holdout_command, "latest_price_date", lambda: weekdays()[-1])
    use_designs(monkeypatch, WIDE)
    frozen = Design(WIDE, target=PLAIN, research_run_key="return-wf1")
    monkeypatch.setitem(production.DESIGNS, "return", frozen)
    return run_panel


def daily_errors(run_key: str, chosen: list[Fold], panel: Path, held: bool) -> pd.DataFrame:
    """Per day, the mean absolute error of the model and of the zero forecast, by hand."""
    frame = target_frame(run_key, chosen, panel, holdout=held)
    rows = frame[frame["in_universe"].astype(bool) & frame["untradable_t1"].eq(False)]
    rows = settled(rows, 5) if held else rows
    rows = rows.dropna(subset=[target_eval.LABEL, target_eval.RETURN_COLUMN])
    label = rows[target_eval.LABEL].astype("float64")
    model = (label - rows[target_eval.RETURN_COLUMN].astype("float64")).abs()
    return pd.DataFrame({"model": model, "zero": label.abs()}).groupby(rows["date"]).mean()


def fold_gaps(daily: pd.DataFrame, chosen: list[Fold]) -> pd.Series:
    days = pd.Index(daily.index)
    inside = {f.index: daily[(days >= f.test_start) & (days <= f.eval_end)] for f in chosen}
    return pd.Series(
        {k: float((rows["model"] - rows["zero"]).mean()) for k, rows in inside.items()}
    )


@pytest.mark.django_db
def test_two_holdout_folds_pool_the_gate_and_weigh_the_flag_by_fold(
    wide: Path, tmp_path: Path
) -> None:
    calendar = TradingDays.from_db()
    every = generate_folds(calendar, weekdays()[-1], WIDE)
    research = [fold for fold in every if fold.research]
    held = [fold for fold in every if fold.holdout]
    design = production.DESIGNS["return"]
    baseline_runs.run_folds("return", research, "return-wf1", wide, target=design.target)
    lgbm_runs.run_folds("return", research, "return-wf1", wide, target=design.target)
    report = tmp_path / "return.md"

    call_command("holdout", scope="return", run_key="return-final", output=str(report))

    bounds = holdout_eval.research_bounds("return", "return-wf1", every, wide, False)
    header = Header("return", "return-final", "return-wf1", 1, weekdays()[-1], rehearsal=False)
    verdict, _ = holdout_eval.judge(header, held, bounds, wide, False, calendar)
    now, past = (
        daily_errors("return-final", held, wide, True),
        daily_errors("return-wf1", research, wide, False),
    )
    gate, raised = verdict.gates[0], verdict.flags[0]
    gaps, earlier = fold_gaps(now, held), fold_gaps(past, research)
    pooled = float((now["model"] - now["zero"]).mean())
    assert (len(research), len(held)) == (2, 2)
    assert gate.value == pytest.approx(float(now["model"].mean()))
    assert gate.passed is bool(now["model"].mean() <= now["zero"].mean())
    assert abs(float(gaps.mean()) - pooled) > 1e-9
    assert raised.holdout == pytest.approx(float(gaps.mean()))
    assert float(earlier.max()) != float(earlier.min())
    assert (raised.high.mean, raised.low.mean) == pytest.approx((earlier.max(), earlier.min()))
    worst = int(str(earlier.idxmax()))
    assert (raised.high.first, raised.high.last) == (worst, worst)
    assert raised.flagged is bool(gaps.mean() > earlier.max())
    text = report.read_text()
    assert f"| {earlier.max():.5f} | 研究摺 {worst}–{worst} |" in text
    expected = FAILED if not gate.passed else DEGRADED if raised.flagged else PASSED
    assert text.splitlines()[4] == f"## 結論：{expected}"
