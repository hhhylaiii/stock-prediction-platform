import math
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import IntegrityError, transaction

from apps.market.models import IndexDaily
from apps.research import candidates, lgbm_runs
from apps.research.baseline_runs import RANK_LABEL, fold_inputs
from apps.research.baselines import fit_ridge
from apps.research.candidates import (
    RELATIVE,
    file_sha256,
    first_signal,
    freeze,
    load_artifact,
    measure,
    relative_labels,
)
from apps.research.holdout import HoldoutError
from apps.research.lock import ResearchBusyError
from apps.research.models import (
    ModelObjective,
    ModelRun,
    ModelStatus,
    Prediction,
    ShadowArtifact,
    ShadowScore,
)
from apps.research.predict import Slot
from apps.research.prediction_cache import fold_record
from apps.research.production import DESIGNS, Design, production_fold
from apps.research.runs import RunError
from apps.research.target_models import RETURN_LABEL, TargetVariant
from apps.research.training import load_labels
from apps.research.walkforward import Fold
from tests.research.run_panels import (
    CONFIG,
    SCREENED,
    claim_all,
    panel_calendar,
    store_taiex,
    use_designs,
)

pytestmark = pytest.mark.django_db
AS_OF = date(2016, 11, 30)
RANK_SLOT = Slot(ModelObjective.RANK, 5)
RANK_RUN = "rank-2016-11.lightgbm.h5.f000"


def _fold() -> Fold:
    return production_fold(panel_calendar(), AS_OF, CONFIG)


def _serving(run_key: str, objective: str, features: list[str], **fold: Any) -> ModelRun:
    chosen = _fold()
    return ModelRun.objects.create(
        run_id=f"{run_key}.lightgbm.h5.f000",
        run_key=run_key,
        model_type="lightgbm",
        objective=objective,
        version="4",
        train_start=chosen.train_start,
        train_end=chosen.train_end,
        params={
            "horizon": 5,
            "fold": {**fold_record(chosen), **fold},
            "bags": 5,
            "production": {"role": "model", "as_of": AS_OF.isoformat()},
        },
        feature_list=features,
        lib_versions={},
        feature_hash="0" * 64,
        status=ModelStatus.ACTIVE,
    )


def _return_features() -> list[str]:
    design = DESIGNS["return"]
    return lgbm_runs.design_features("return", design.variant, design.target)


@pytest.fixture
def served(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Claimed holdouts, TAIEX_TR closes and serving rank and return runs on one production fold."""
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 60)
    monkeypatch.setattr(candidates, "BAGS", 2)
    store_taiex()
    use_designs(
        monkeypatch, CONFIG, **{"return": Design(CONFIG, target=TargetVariant(market=False))}
    )
    claim_all()
    _serving("rank-2016-11", ModelObjective.RANK, ["tech_ret_1d"])
    _serving("return-2016-11", ModelObjective.REGRESSION, _return_features())
    return run_panel


def _levels() -> dict[date, float]:
    rows = IndexDaily.objects.filter(index_id="TAIEX_TR")
    return {day: float(close) for day, close in rows.values_list("date", "close")}


def test_relative_labels_take_taiex_tr_s_five_sessions_off_the_open_return(served: Path) -> None:
    calendar = panel_calendar()
    first, last = date(2016, 3, 1), date(2016, 3, 8)
    labels = pd.DataFrame({"date": [first, last], RETURN_LABEL: [0.05, 0.02]})
    level = _levels()
    expected = 0.05 - (level[calendar.shift(first, 5)] / level[first] - 1)
    assert relative_labels(labels, calendar)[RETURN_LABEL].iloc[0] == pytest.approx(expected)
    IndexDaily.objects.filter(index_id="TAIEX_TR", date=calendar.shift(last, 5)).delete()
    assert math.isnan(relative_labels(labels, calendar)[RETURN_LABEL].iloc[1])


def test_c1_freezes_the_rank_ridge_of_the_serving_fold(served: Path) -> None:
    row = freeze("C1", served)
    fold = _fold()
    assert (row.generation, row.as_of, row.serving_run_id) == (
        1,
        AS_OF,
        RANK_RUN,
    )
    assert (row.train_start, row.train_end, row.fold) == (
        fold.train_start,
        fold.train_end,
        fold_record(fold),
    )
    assert row.path == "shadow/c1/rank-2016-11/g1.pkl"
    path = Path(settings.MODEL_DIR) / row.path
    assert row.sha256 == file_sha256(path)
    assert row.first_signal == panel_calendar().shift(AS_OF, 1)
    assert (row.recipe["model"], row.recipe["label"], row.recipe["features"]) == (
        "ridge",
        RANK_LABEL,
        SCREENED,
    )
    read = fold_inputs(fold, fold.train_start, fold.train_end, served, SCREENED)
    train = read.train(load_labels(fold.train_start, fold.train_end, 5), RANK_LABEL)
    direct = fit_ridge(train, SCREENED, RANK_LABEL, 5)
    frozen = load_artifact(row)
    assert frozen.penalty == direct.penalty == row.recipe["penalty"]
    np.testing.assert_allclose(frozen.predict(train), direct.predict(train))


def test_c2_fits_the_return_design_on_the_relative_label(
    served: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[pd.DataFrame] = []
    fit = candidates.fit_return

    def spy(inputs: pd.DataFrame, labels: pd.DataFrame, *args: Any) -> Any:
        seen.append(labels)
        return fit(inputs, labels, *args)

    monkeypatch.setattr(candidates, "fit_return", spy)
    row = freeze("C2", served)
    fold = _fold()
    absolute = load_labels(fold.train_start, fold.train_end, 5)
    level, calendar = _levels(), panel_calendar()
    market = absolute["date"].map(lambda day: level[calendar.shift(day, 5)] / level[day] - 1)
    np.testing.assert_allclose(seen[0][RETURN_LABEL], absolute[RETURN_LABEL] - market)
    model = load_artifact(row)
    assert sorted(model) == ["booster", "market_bins"]
    assert len(model["booster"].boosters) == 2
    assert (row.recipe["label"], row.recipe["bags"], row.recipe["features"]) == (
        RELATIVE,
        2,
        _return_features(),
    )
    assert row.path == "shadow/c2/return-2016-11/g1.pkl"


def test_measuring_writes_no_row_and_the_same_data_refits_to_the_same_file(
    served: Path, tmp_path: Path
) -> None:
    for name in ("C1", "C2"):
        once = measure(name, served, tmp_path / "a")
        again = measure(name, served, tmp_path / "b")
        assert once.sha256 == again.sha256 == file_sha256(once.path)
        assert once.seconds >= 0
        assert once.peak_mb > 0
    assert not ShadowArtifact.objects.exists()
    assert not (Path(settings.MODEL_DIR) / "shadow").exists()


def test_a_second_freeze_on_the_same_run_needs_replace_and_keeps_the_first(served: Path) -> None:
    first = freeze("C1", served)
    with pytest.raises(RunError, match="replace writes generation 2"):
        freeze("C1", served)
    second = freeze("C1", served, replace=True)
    assert (second.generation, second.path) == (2, "shadow/c1/rank-2016-11/g2.pkl")
    kept = ShadowArtifact.objects.get(candidate="C1", generation=1)
    assert (kept.sha256, kept.path) == (first.sha256, first.path)
    assert (Path(settings.MODEL_DIR) / first.path).exists()


def test_the_first_signal_follows_the_latest_written_prediction(served: Path) -> None:
    calendar = panel_calendar()
    assert first_signal(RANK_SLOT, AS_OF, calendar) == calendar.shift(AS_OF, 1)
    later = calendar.shift(AS_OF, 3)
    Prediction.objects.create(
        run_id=RANK_RUN,
        stock_id="1100",
        date=later,
        horizon=5,
        score=0.1,
        in_universe=True,
    )
    assert first_signal(RANK_SLOT, AS_OF, calendar) == calendar.shift(later, 1)
    assert freeze("C1", served).first_signal == calendar.shift(later, 1)


def test_a_candidate_without_a_serving_run_is_refused(served: Path) -> None:
    ModelRun.objects.filter(objective=ModelObjective.REGRESSION).update(status=ModelStatus.RETIRED)
    with pytest.raises(RunError, match="no serving run for regression 5 日"):
        freeze("C2", served)


def test_only_a_production_fold_is_fit_on(served: Path) -> None:
    ModelRun.objects.filter(run_id=RANK_RUN).delete()
    _serving("rank-2016-11", ModelObjective.RANK, ["tech_ret_1d"], holdout=True)
    with pytest.raises(HoldoutError, match="production folds only"):
        freeze("C1", served)
    assert not ShadowArtifact.objects.exists()


def test_a_serving_run_without_a_production_note_is_refused(served: Path) -> None:
    run = ModelRun.objects.get(run_id=RANK_RUN)
    run.params = {key: value for key, value in run.params.items() if key != "production"}
    run.save()
    with pytest.raises(RunError, match="not a production model N"):
        freeze("C1", served)


def test_c2_refuses_a_serving_run_on_other_features(served: Path) -> None:
    ModelRun.objects.filter(objective=ModelObjective.REGRESSION).update(
        feature_list=["tech_ret_1d"]
    )
    with pytest.raises(RunError, match="other features than the frozen return design"):
        freeze("C2", served)


def test_a_busy_research_lock_fits_and_writes_nothing(
    served: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def busy(holder: str) -> Iterator[None]:
        raise ResearchBusyError("held by build_features")
        yield

    fitted: list[str] = []
    monkeypatch.setattr(candidates, "research_lock", busy)
    monkeypatch.setattr(candidates, "serving", lambda chosen: fitted.append("fit"))
    with pytest.raises(ResearchBusyError):
        freeze("C1", served)
    assert fitted == []
    assert not ShadowArtifact.objects.exists()


def test_a_failed_insert_leaves_no_model_file(
    served: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(**fields: Any) -> None:
        raise IntegrityError("shadow_artifact_sha256")

    monkeypatch.setattr(ShadowArtifact.objects, "create", refuse)
    with pytest.raises(IntegrityError):
        freeze("C1", served)
    folder = Path(settings.MODEL_DIR) / "shadow" / "c1" / "rank-2016-11"
    assert list(folder.iterdir()) == []


def test_a_missing_or_changed_file_is_refused(served: Path) -> None:
    row = freeze("C1", served)
    path = Path(settings.MODEL_DIR) / row.path
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(RunError, match="no longer matches its frozen sha256"):
        load_artifact(row)
    path.unlink()
    with pytest.raises(RunError, match=r"has no file at shadow/c1/rank-2016-11/g1\.pkl"):
        load_artifact(row)


def _artifact(**fields: Any) -> ShadowArtifact:
    run = ModelRun.objects.get(run_id=RANK_RUN)
    fold = _fold()
    values = {
        "candidate": "C1",
        "serving_run": run,
        "generation": 1,
        "as_of": AS_OF,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "fold": fold_record(fold),
        "path": "shadow/c1/rank-2016-11/g1.pkl",
        "sha256": "a" * 64,
        "recipe": {},
        "first_signal": date(2016, 12, 1),
        **fields,
    }
    return ShadowArtifact(**values)


@pytest.mark.parametrize(
    "fields",
    [
        {"generation": 0},
        {"sha256": "A" * 64},
        {"sha256": "a" * 63},
        {"first_signal": AS_OF},
        {"train_start": date(2016, 11, 1), "train_end": date(2016, 11, 1)},
    ],
)
def test_the_artifact_table_refuses_a_bad_row(served: Path, fields: dict[str, Any]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        _artifact(**fields).save()


@pytest.mark.parametrize(
    "fields",
    [
        {"pct": 0.5, "in_universe": False},
        {"pct": 0.5, "score": None},
        {"pct": 1.5},
        {"score": math.nan},
        {"raw": math.inf},
    ],
)
def test_the_score_table_refuses_a_bad_row(served: Path, fields: dict[str, Any]) -> None:
    run = ModelRun.objects.get(run_id=RANK_RUN)
    values = {
        "candidate": "C1",
        "date": date(2016, 12, 1),
        "stock_id": "1100",
        "serving_run": run,
        "generation": 1,
        "score": 0.4,
        "raw": 0.1,
        "pct": 0.5,
        "in_universe": True,
        **fields,
    }
    with pytest.raises(IntegrityError), transaction.atomic():
        ShadowScore.objects.create(**values)


def test_the_command_freezes_or_only_measures(served: Path, tmp_path: Path) -> None:
    out = StringIO()
    call_command("shadow_train", "--candidate", "C1", "--output", str(tmp_path), stdout=out)
    assert "C1: " in out.getvalue()
    assert not ShadowArtifact.objects.exists()
    out = StringIO()
    call_command("shadow_train", "--candidate", "C1", stdout=out)
    row = ShadowArtifact.objects.get()
    assert f"C1 g1 on {row.serving_run_id}: {row.path} sha256 {row.sha256}" in out.getvalue()
    with pytest.raises(CommandError, match="replace writes generation 2"):
        call_command("shadow_train", "--candidate", "C1", stdout=StringIO())
    with pytest.raises(CommandError, match="--replace freezes"):
        call_command("shadow_train", "--candidate", "C1", "--replace", "--output", str(tmp_path))
