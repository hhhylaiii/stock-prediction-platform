import math
from datetime import date
from typing import cast

import numpy as np
import pandas as pd
import pytest

from apps.research.eval_report import rank_metrics, render_report, write_rank_metrics
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
    """A weak 5-day score against a noise baseline, and a 20-day score that copies its label."""
    frame = panel()
    rng = np.random.default_rng(3)
    weak = frame.assign(
        score=frame["fwd_ret_open"] + rng.normal(0, 1.0, len(frame)),
        noise=rng.normal(size=len(frame)),
    )
    leaked = frame.assign(score=frame["fwd_rank_pct"])
    return (
        rank_report(weak, "score", 5, folds(), baseline="noise"),
        rank_report(leaked, "score", 20, folds()),
    )


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
    assert "| 1 | a, b | — |" in text
    assert "| 2 | b, c | 1 |" in text
    top = cast(pd.Series, deciles[5].loc[("all", "D10")])
    assert (
        f"| 全部 | D10 | {top['monthly_return'] * 100:.2f}% | {top['monthly_excess'] * 100:.2f}% |"
        in text
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
    assert total.loc["rank_ic", "value"] == pytest.approx(five.total.mean)
    assert total.loc["rank_ic_nw_t", "value"] == pytest.approx(five.total.nw_t)
    assert total.loc["beat_baseline_share", "value"] == five.beat_share
    assert total.loc["beat_baseline_folds", "value"] == five.compared_folds
    assert total.loc["rank_ic", "period_start"] == research()[0].test_start
    assert total.loc["rank_ic", "period_end"] == research()[-1].eval_end
    folds_ic = rows[(rows["grain"] == "fold") & (rows["metric"] == "rank_ic")]
    assert folds_ic["period_start"].tolist() == [fold.test_start for fold in research()]
    years = rows[(rows["grain"] == "year") & (rows["metric"] == "rank_ic")]
    assert years["period_start"].tolist() == [date(2016, 1, 1), date(2017, 1, 1)]
    assert years["period_end"].tolist() == [date(2016, 12, 31), date(2017, 12, 31)]
    months = rows[(rows["grain"] == "month") & (rows["metric"] == "rank_ic")]
    assert months["value"].tolist() == pytest.approx(five.monthly.tolist())
    assert months["period_end"].iloc[1] == date(2016, 10, 31)
    liquidity = rows[rows["segment"] != ""]
    assert sorted(set(liquidity["segment"])) == ["liq_t1", "liq_t2", "liq_t3"]
    top = liquidity[(liquidity["segment"] == "liq_t3") & (liquidity["metric"] == "rank_ic")]
    assert top["value"].tolist() == pytest.approx([five.by_liquidity.loc[3, "mean"]])
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
    BacktestMetric.objects.create(model_type="lightgbm", **other)
    BacktestMetric.objects.create(model_type="ridge", **other)
    BacktestMetric.objects.create(model_type="lightgbm", **{**other, "horizon": 20})
    BacktestMetric.objects.create(model_type="lightgbm", **{**other, "objective": "regression"})
    BacktestMetric.objects.create(model_type="lightgbm", **{**other, "run_key": "rank-y"})

    written = write_rank_metrics(five, "rank-x", "lightgbm")
    again = write_rank_metrics(five, "rank-x", "lightgbm")

    stored = BacktestMetric.objects.filter(
        run_key="rank-x", model_type="lightgbm", objective="rank", horizon=5
    )
    assert written == again == stored.count() == len(rank_metrics(five, "rank-x", "lightgbm"))
    assert not stored.filter(period_start=date(2010, 1, 1)).exists()
    assert BacktestMetric.objects.filter(model_type="ridge").count() == 1
    assert BacktestMetric.objects.filter(period_start=date(2010, 1, 1)).count() == 4
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
