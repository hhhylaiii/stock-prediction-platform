import math
from datetime import date
from typing import cast

import numpy as np
import pandas as pd
import pytest

from apps.research.eval_report import (
    RANK_METRICS,
    rank_metrics,
    render_report,
    write_rank_metrics,
)
from apps.research.evaluation import (
    RankReport,
    consistency_report,
    coverage_report,
    decile_report,
    importance_drift,
    rank_report,
)
from apps.research.models import BacktestMetric
from tests.research.test_evaluation import calendar, folds, inside, panel, prices, research


def reports() -> tuple[RankReport, RankReport]:
    """A weak 5-day score against a baseline that wins one fold, and a 20-day leaked score."""
    frame = panel()
    rng = np.random.default_rng(3)
    weak = frame.assign(
        score=frame["fwd_ret_open"] + rng.normal(0, 1.0, len(frame)),
        noise=rng.normal(size=len(frame)),
    )
    won = inside(weak, [research()[0]])
    weak.loc[won, "noise"] = weak.loc[won, "fwd_rank_pct"]
    leaked = frame.assign(score=frame["fwd_rank_pct"])
    return (
        rank_report(weak, "score", 5, folds(), baseline="noise"),
        rank_report(leaked, "score", 20, folds()),
    )


def assert_tables_are_rectangular(text: str) -> None:
    """Every row of each Markdown table holds as many cells as its header."""
    widths: list[int] = []
    for line in [*text.splitlines(), ""]:
        if line.startswith("|"):
            widths.append(line.count("|"))
            continue
        assert len(set(widths)) <= 1, widths
        widths = []


def test_report_renders_every_section_from_a_synthetic_prediction_set() -> None:
    frame = panel()
    five, twenty = reports()
    deciles = {5: decile_report(frame, "score", 5, folds(), prices(frame), calendar())}
    drift = importance_drift({1: {"a": 2.0, "b": 1.0}, 2: {"b": 3.0, "c": 1.0}})
    coverage = coverage_report(frame, "fwd_ret_open", "q10", "q90", folds())
    consistency = consistency_report(frame, "q10", "q50", "q90", "vol", "prob", folds())

    text = render_report(
        "rank-x lightgbm",
        [five, twenty],
        deciles=deciles,
        importance=drift,
        coverage=coverage,
        consistency=consistency,
    )

    for heading in (
        "# 評估報表：rank-x lightgbm",
        "## 排名 5 日",
        "## 排名 20 日",
        "### 每摺",
        "### 每年",
        "### 成交金額三分位",
        "### 十分位（扣成本）",
        "## 特徵重要度前 10 名",
        "## 區間覆蓋率（tech_vol_20d 五分位）",
        "## 輸出一致性",
    ):
        assert heading in text
    assert f"{five.total.mean:.4f}" in text
    research_folds = sum(fold.research for fold in five.folds)
    assert (
        f"贏 baseline 的研究摺比例：{five.beat_share:.0%}"
        f"（{five.compared_folds}／{research_folds} 摺可比）" in text
    )
    assert "| 共同列 IC | baseline IC | 贏 baseline |" in text
    first = five.by_fold.loc[research()[0].index]
    verdict = "是" if first["beats"] else "否"
    tail = f"| {first['paired_mean']:.4f} | {first['baseline_mean']:.4f} | {verdict} |"
    assert any(
        line.startswith(f"| {research()[0].index} | 研究 ") and line.endswith(tail)
        for line in text.splitlines()
    )
    assert text.count("🔴 洩漏警訊") == 1
    assert "20 日研究期 IC 均值 1.0000 高於 0.15" in text
    assert "| D10-D1 |" in text
    assert "成交金額前三分之一" in text
    total, neutral = five.total, five.neutralized
    assert (
        f"- 研究期 IC 均值 {total.mean:.4f}，標準差 {total.std:.4f}，NW t {total.nw_t:.2f}，"
        f"ICIR {total.icir:.2f}，{total.n_days} 天" in text
    )
    assert f"- IC 為正的研究摺比例：{five.positive_share:.0%}\n" in text
    assert f"- 中性化 IC：{neutral.mean:.4f}，NW t {neutral.nw_t:.2f}，{neutral.n_days} 天" in text
    assert "| IC 為正的天數比例 | 天數 |" in text
    year = cast(pd.Series, five.by_year.loc[2016])
    assert (
        f"| 2016 | {year['mean']:.4f} | {year['std']:.4f} | {year['icir']:.2f} "
        f"| {year['nw_t']:.2f} | {year['positive_day_share']:.0%} | {int(year['n_days'])} |"
    ) in text
    high = cast(pd.Series, five.by_liquidity.loc[3])
    assert f"| 高 | {high['mean']:.4f} | {high['std']:.4f} |" in text
    tuning = next(fold for fold in folds() if fold.tuning)
    holdout = next(fold for fold in folds() if fold.holdout)
    lines = text.splitlines()
    assert any(line.startswith(f"| {tuning.index} | 調參 | {tuning.test_start}") for line in lines)
    assert any(
        line.startswith(f"| {holdout.index} | holdout | {holdout.test_start}") for line in lines
    )
    assert "| 1 | a, b | — |" in text
    twenty_section = text.split("## 排名 20 日")[1].split("## 特徵重要度")[0]
    assert "贏 baseline" not in twenty_section
    calm = cast(pd.Series, coverage.loc[1])
    assert f"| 1 | {calm['coverage'] * 100:.1f}% | {calm['width']:.4f} |" in text
    assert coverage.index.tolist() == [1, 2, 3, 4, 5, "all"]
    assert all(f"| {bucket} | " in text for bucket in (2, 3, 4, 5))
    assert_tables_are_rectangular(text)
    assert "| 2 | b, c | 1 |" in text
    top = cast(pd.Series, deciles[5].loc[("all", "D10")])
    assert (
        f"| 全部 | D10 | {top['monthly_return'] * 100:.2f}% | {top['monthly_excess'] * 100:.2f}% "
        f"| {top['excess_drawdown'] * 100:.2f}% | {top['annual_turnover']:.2f} "
        f"| {int(top['n_days'])} |" in text
    )
    whole = cast(pd.Series, coverage.loc["all"])
    assert f"| 全體 | {whole['coverage'] * 100:.1f}% | {whole['width']:.4f} |" in text
    odds = cast(pd.Series, consistency.loc["contradiction"])
    assert (
        f"| 上漲機率與 q50 矛盾 | {odds['mean_share'] * 100:.1f}% | {odds['max_share'] * 100:.1f}% "
        f"| {int(odds['days_over'])} | {int(odds['n_days'])} |" in text
    )
    assert "| 寬度比超出 0.5 到 2 |" in text


def test_report_skips_sections_it_was_not_given() -> None:
    five, _ = reports()

    text = render_report("rank-x ridge", [five])

    assert "## 排名 5 日" in text
    assert "十分位" not in text
    assert "特徵重要度" not in text
    assert "覆蓋率" not in text
    assert "一致性" not in text
    assert "🔴" not in text


def test_rank_metrics_hold_total_fold_year_month_and_liquidity_rows() -> None:
    five, _ = reports()

    rows = rank_metrics(five, "rank-x", "lightgbm")

    assert set(rows["run_key"]) == {"rank-x"}
    assert set(rows["model_type"]) == {"lightgbm"}
    assert set(rows["objective"]) == {"rank"}
    assert set(rows["horizon"]) == {5}
    total = rows[(rows["grain"] == "total") & (rows["segment"] == "")].set_index("metric")
    expected = {
        "rank_ic": five.total.mean,
        "rank_icir": five.total.icir,
        "rank_ic_nw_t": five.total.nw_t,
        "positive_fold_share": five.positive_share,
        "beat_baseline_share": five.beat_share,
        "beat_baseline_folds": float(five.compared_folds),
        "neutral_rank_ic": five.neutralized.mean,
        "neutral_rank_ic_nw_t": five.neutralized.nw_t,
    }
    assert five.positive_share != five.beat_share
    assert total["value"].to_dict() == pytest.approx(expected)
    assert total.loc["rank_ic", "period_start"] == research()[0].test_start
    assert total.loc["rank_ic", "period_end"] == research()[-1].eval_end
    chosen = [fold.index for fold in research()]
    for metric, column in (("rank_ic", "mean"), ("rank_ic_nw_t", "nw_t")):
        by_fold = rows[(rows["grain"] == "fold") & (rows["metric"] == metric)]
        assert by_fold["period_start"].tolist() == [fold.test_start for fold in research()]
        assert by_fold["period_end"].tolist() == [fold.eval_end for fold in research()]
        assert by_fold["value"].tolist() == pytest.approx(five.by_fold.loc[chosen, column].tolist())
        years = rows[(rows["grain"] == "year") & (rows["metric"] == metric)]
        assert years["period_start"].tolist() == [date(2016, 1, 1), date(2017, 1, 1)]
        assert years["period_end"].tolist() == [date(2016, 12, 31), date(2017, 12, 31)]
        assert years["value"].tolist() == pytest.approx(five.by_year[column].tolist())
        terciles = rows[(rows["segment"] != "") & (rows["metric"] == metric)]
        assert terciles["segment"].tolist() == ["liq_t1", "liq_t2", "liq_t3"]
        assert set(terciles["grain"]) == {"total"}
        assert set(terciles["period_start"]) == {research()[0].test_start}
        assert set(terciles["period_end"]) == {research()[-1].eval_end}
        assert terciles["value"].tolist() == pytest.approx(five.by_liquidity[column].tolist())
    months = rows[(rows["grain"] == "month") & (rows["metric"] == "rank_ic")]
    assert months["value"].tolist() == pytest.approx(five.monthly.tolist())
    assert months["period_start"].tolist() == five.monthly.index.tolist()
    assert months["period_end"].iloc[1] == date(2016, 10, 31)
    days = rows[(rows["grain"] == "month") & (rows["metric"] == "rank_ic_days")]
    assert days["period_start"].tolist() == five.monthly_days.index.tolist()
    assert days["value"].tolist() == five.monthly_days.astype(float).tolist()
    assert set(rows["metric"]) == set(RANK_METRICS)
    assert rows.duplicated(["metric", "grain", "period_start", "period_end", "segment"]).sum() == 0


@pytest.mark.django_db
def test_writing_rank_metrics_replaces_the_group_and_keeps_others() -> None:
    five, _ = reports()
    other = {
        "run_key": "rank-x",
        "objective": "rank",
        "horizon": 5,
        "metric": "rank_ic",
        "grain": "month",
        "period_start": date(2010, 1, 1),
        "period_end": date(2010, 1, 31),
        "value": 0.5,
    }
    for name in RANK_METRICS:
        BacktestMetric.objects.create(model_type="lightgbm", **{**other, "metric": name})
    BacktestMetric.objects.create(model_type="ridge", **other)
    BacktestMetric.objects.create(model_type="lightgbm", **{**other, "horizon": 20})
    BacktestMetric.objects.create(model_type="lightgbm", **{**other, "objective": "regression"})
    BacktestMetric.objects.create(model_type="lightgbm", **{**other, "run_key": "rank-y"})
    BacktestMetric.objects.create(model_type="lightgbm", **{**other, "metric": "decile_spread"})

    written = write_rank_metrics(five, "rank-x", "lightgbm")
    again = write_rank_metrics(five, "rank-x", "lightgbm")

    stored = BacktestMetric.objects.filter(
        run_key="rank-x",
        model_type="lightgbm",
        objective="rank",
        horizon=5,
        metric__in=RANK_METRICS,
    )
    assert written == again == stored.count() == len(rank_metrics(five, "rank-x", "lightgbm"))
    assert not stored.filter(period_start=date(2010, 1, 1)).exists()
    assert BacktestMetric.objects.filter(model_type="ridge").count() == 1
    assert BacktestMetric.objects.filter(period_start=date(2010, 1, 1)).count() == 5
    assert not BacktestMetric.objects.filter(
        model_type="lightgbm",
        horizon=5,
        objective="rank",
        run_key="rank-x",
        metric__in=RANK_METRICS,
        period_start=date(2010, 1, 1),
    ).exists()
    assert BacktestMetric.objects.filter(metric="decile_spread").count() == 1
    beat = stored.get(metric="beat_baseline_share", grain="total")
    assert beat.value == pytest.approx(five.beat_share)


@pytest.mark.django_db
def test_metrics_without_a_value_are_stored_empty() -> None:
    _, twenty = reports()
    assert math.isnan(twenty.beat_share)

    write_rank_metrics(twenty, "rank-x", "lightgbm")

    assert BacktestMetric.objects.get(metric="beat_baseline_share", grain="total").value is None
    assert BacktestMetric.objects.get(metric="beat_baseline_folds", grain="total").value is None
    total = BacktestMetric.objects.get(metric="rank_ic", grain="total", segment="")
    assert total.value == pytest.approx(twenty.total.mean)


def test_a_fold_without_a_baseline_verdict_renders_empty() -> None:
    frame = panel()
    frame["noise"] = np.random.default_rng(5).normal(size=len(frame))
    frame.loc[inside(frame, [research()[3]]), "noise"] = np.nan
    report = rank_report(frame, "score", 5, folds(), baseline="noise")

    text = render_report("rank-x lightgbm", [report])

    row = next(
        line for line in text.splitlines() if line.startswith(f"| {research()[3].index} | 研究 ")
    )
    assert row.endswith("| — | — | — |")
    assert "（3／4 摺可比）" in text


def test_percent_cells_without_a_value_render_a_bare_dash() -> None:
    frame = panel().assign(q10=np.nan, prob=np.nan)
    coverage = coverage_report(frame, "fwd_ret_open", "q10", "q90", folds())
    consistency = consistency_report(frame, "q10", "q50", "q90", "vol", "prob", folds())

    blank = rank_report(frame.assign(score=np.nan), "score", 5, folds())

    text = render_report("rank-x lightgbm", [blank], coverage=coverage, consistency=consistency)

    assert "| 全體 | — | — |" in text
    assert "| 上漲機率與 q50 矛盾 | — | — | 0 | 0 |" in text
    assert "—%" not in text
    assert "- IC 為正的研究摺比例：—\n" in text
    assert "nan" not in text
