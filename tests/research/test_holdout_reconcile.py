"""The rehearsal's reconciliation with the research run's own fit of the same fold."""

from io import StringIO
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.research import holdout_reconcile
from apps.research.holdout_reconcile import Matched
from tests.research.run_panels import folds
from tests.research.test_holdout_eval import frozen, researched  # noqa: F401

ROWS = {
    "rank": [
        "判定：排名 5 日 IC",
        "判定：排名 20 日 IC",
        "旗標：排名 5 日 IC",
        "旗標：排名 20 日 IC",
    ],
    "return": ["判定：5 日 MAE", "旗標：MAE 差（模型減零預測）"],
    "interval": [
        "判定：20 日總覆蓋率",
        "判定：機率 Brier",
        "判定：波動率 QLIKE",
        "旗標：pinball 差（模型減波動率區間）",
        "旗標：Brier 差（模型減 logistic）",
        "旗標：20 日覆蓋率",
        "另算：pinball（模型）",
        "另算：日均 AUC",
    ],
}


def rehearse(scope: str, out: StringIO | None = None, **options: object) -> str:
    out = StringIO() if out is None else out
    call_command(
        "holdout", scope=scope, run_key=f"{scope}-rehearsal", rehearse=True, stdout=out, **options
    )
    return out.getvalue()


def test_a_row_agrees_within_the_tolerance_and_never_on_a_missing_value() -> None:
    assert Matched("m", 0.12345, 0.12349).agrees
    assert not Matched("m", 0.1234, 0.1235).agrees
    assert not Matched("m", float("nan"), float("nan")).agrees


@pytest.mark.django_db
@pytest.mark.parametrize("scope", list(ROWS))
def test_a_rehearsal_on_the_research_seed_reconciles_with_the_research_run(
    frozen: Path,  # noqa: F811
    scope: str,
) -> None:
    researched(scope, frozen)
    last = max(fold.index for fold in folds() if fold.research)

    printed = rehearse(scope)

    assert f"## 對帳：演練摺 {last} 對研究 run `{scope}-wf1` 的同一摺" in printed
    assert "容許誤差 0.00005" in printed
    for name in ROWS[scope]:
        assert f"| {name} |" in printed
    assert printed.count("| 相符 |") == len(ROWS[scope])
    assert "對帳結論：相符" in printed
    assert "不符" not in printed


@pytest.mark.django_db
def test_a_rehearsal_that_does_not_reconcile_fails_after_printing_its_report(
    frozen: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    researched("return", frozen)
    monkeypatch.setattr(holdout_reconcile, "TOLERANCE", -1.0)
    out, report = StringIO(), tmp_path / "rehearsal.md"

    with pytest.raises(CommandError, match=r"does not reconcile with return-wf1.*5 日 MAE"):
        rehearse("return", out, output=str(report))

    assert "對帳結論：不符，不得進正式跑" in report.read_text()
    assert report.read_text().count("| 不符 |") == len(ROWS["return"])


@pytest.mark.django_db
def test_a_rehearsal_on_other_bags_is_not_reconciled(
    frozen: Path,  # noqa: F811
) -> None:
    researched("return", frozen)

    printed = rehearse("return", bags=2)

    assert "## 對帳：未對帳" in printed
    assert "演練的 (seed, bags) 是 [(0, 2)]，研究 run 這一摺是 [(0, 1)]" in printed
    assert "對帳結論" not in printed
