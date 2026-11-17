import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
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
from apps.research.candidate_scores import score_candidate
from apps.research.candidates import (
    RELATIVE,
    file_sha256,
    first_signal,
    freeze,
    load_artifact,
    measure,
    relative_labels,
    unfollowed,
)
from apps.research.features.store import columns_for, read_partitions
from apps.research.gbdt import bag_seeds
from apps.research.holdout import HoldoutError
from apps.research.inference import day_rows, model_inputs
from apps.research.lock import ResearchBusyError
from apps.research.models import (
    FeatureDaily,
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
from apps.research.runs import RunError, drop_run
from apps.research.target_models import RETURN_LABEL, TargetVariant
from apps.research.training import linear_inputs, load_labels
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
TODAY = date(2016, 12, 1)
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
    monkeypatch.setattr(candidates.timezone, "localdate", lambda: TODAY)
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


def test_a_cross_validation_score_without_a_value_is_kept_as_null(
    served: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = candidates.fit_ridge

    def gapped(*args: Any) -> Any:
        fitted = fit(*args)
        return replace(fitted, scores={**fitted.scores, 0.1: math.nan})

    monkeypatch.setattr(candidates, "fit_ridge", gapped)
    row = freeze("C1", served)
    row.refresh_from_db()
    assert row.recipe["cv"]["0.1"] is None


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
    assert (row.recipe["lightgbm"]["seed"], row.recipe["bag_seeds"]) == (0, bag_seeds(0, 2))
    params = row.recipe["lightgbm"]
    assert (params["objective"], params["metric"], params["alpha"]) == (
        "huber",
        "huber",
        row.recipe["delta"],
    )
    assert row.path == "shadow/c2/return-2016-11/g1.pkl"


def test_measuring_writes_no_row_and_the_same_data_refits_to_the_same_file(
    served: Path, tmp_path: Path
) -> None:
    first = [measure(name, served, tmp_path / "a") for name in ("C1", "C2")]
    again = [measure(name, served, tmp_path / "b") for name in ("C1", "C2")]
    assert [found.path.name for found in first] == ["c1.pkl", "c2.pkl"]
    assert [found.sha256 for found in first] == [found.sha256 for found in again]
    assert [found.sha256 for found in first] == [file_sha256(found.path) for found in first]
    assert first[0].sha256 != first[1].sha256
    assert all(found.seconds >= 0 for found in first)
    assert all(found.peak_mb > 0 for found in first)
    assert not ShadowArtifact.objects.exists()
    assert not (Path(settings.MODEL_DIR) / "shadow").exists()


def test_only_a_lost_or_changed_file_is_replaced_and_the_first_generation_stays(
    served: Path,
) -> None:
    first = freeze("C1", served)
    with pytest.raises(RunError, match="replace writes generation 2 if its file is lost"):
        freeze("C1", served)
    with pytest.raises(RunError, match="file is intact; replace only rebuilds"):
        freeze("C1", served, replace=True)
    path = Path(settings.MODEL_DIR) / first.path
    changed = path.read_bytes() + b"x"
    path.write_bytes(changed)
    second = freeze("C1", served, replace=True)
    assert (second.generation, second.path) == (2, "shadow/c1/rank-2016-11/g2.pkl")
    kept = ShadowArtifact.objects.get(candidate="C1", generation=1)
    assert (kept.sha256, kept.path) == (first.sha256, first.path)
    assert path.read_bytes() == changed
    assert load_artifact(second).penalty == first.recipe["penalty"]


def test_drop_run_keeps_the_frozen_candidates(served: Path) -> None:
    row = freeze("C1", served)
    with pytest.raises(RunError, match="holds the frozen shadow candidates"):
        drop_run("shadow")
    assert (Path(settings.MODEL_DIR) / row.path).exists()


def test_the_first_signal_follows_the_latest_written_prediction(served: Path) -> None:
    calendar = panel_calendar()
    assert first_signal(RANK_SLOT, AS_OF, calendar, TODAY) == calendar.shift(AS_OF, 1)
    later = calendar.shift(AS_OF, 3)
    Prediction.objects.create(
        run_id=RANK_RUN,
        stock_id="1100",
        date=later,
        horizon=5,
        score=0.1,
        in_universe=True,
    )
    assert first_signal(RANK_SLOT, AS_OF, calendar, TODAY) == calendar.shift(later, 1)
    assert freeze("C1", served).first_signal == calendar.shift(later, 1)


def test_the_first_signal_never_reaches_back_before_the_freeze(served: Path) -> None:
    calendar = panel_calendar()
    assert first_signal(RANK_SLOT, AS_OF, calendar, date(2016, 12, 8)) == date(2016, 12, 8)
    assert first_signal(RANK_SLOT, AS_OF, calendar, date(2016, 12, 10)) == date(2016, 12, 12)


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


def _predicted(days: list[date]) -> None:
    """C0's rank 5-day rows of the days: the panel's universe, scored by tech_ret_1d."""
    Prediction.objects.bulk_create(
        Prediction(
            run_id=RANK_RUN,
            stock_id=row.stock_id,
            date=row.date,
            horizon=5,
            score=row.tech_ret_1d,
            in_universe=row.in_universe,
        )
        for row in FeatureDaily.objects.filter(date__in=days)
    )


def _scores(name: str, day: date) -> pd.DataFrame:
    rows = ShadowScore.objects.filter(candidate=name, date=day).order_by("stock_id")
    fields = ["stock_id", "score", "raw", "pct", "in_universe", "serving_run_id", "generation"]
    return pd.DataFrame(list(rows.values_list(*fields)), columns=fields).set_index("stock_id")


SCORED = [date(2016, 12, 1), date(2016, 12, 2)]


def test_c1_averages_its_ridge_rank_and_c0_s_rank_in_the_universe(served: Path) -> None:
    row = freeze("C1", served)
    _predicted(SCORED)

    got = score_candidate("C1", SCORED[-1], panel_calendar())

    assert got.scored == ["2016-12-01", "2016-12-02"]
    assert got.artifacts == [f"{RANK_RUN} g1"]
    day = SCORED[0]
    read = read_partitions(day, day, served, columns_for(SCREENED))
    inputs = linear_inputs(read)
    ridge = pd.Series(load_artifact(row).predict(inputs), index=inputs["stock_id"].to_numpy())
    c0 = pd.Series({p.stock_id: p.score for p in Prediction.objects.filter(date=day)})
    inside = pd.Series({p.stock_id: p.in_universe for p in Prediction.objects.filter(date=day)})
    ranks = [series.where(inside).rank(pct=True) for series in (ridge, c0)]
    expected = (ranks[0] + ranks[1]) / 2
    stored = _scores("C1", day)
    np.testing.assert_allclose(stored["raw"], ridge.reindex(stored.index))
    universe = stored[stored["in_universe"]]
    np.testing.assert_allclose(universe["score"], expected.reindex(universe.index))
    np.testing.assert_allclose(universe["pct"], universe["score"].rank(pct=True))
    outside = ShadowScore.objects.get(candidate="C1", date=day, stock_id="1139")
    assert (outside.in_universe, outside.score, outside.pct) == (False, None, None)
    assert set(stored["serving_run_id"]) == {RANK_RUN}
    assert set(stored["generation"]) == {1}


def test_c2_scores_the_relative_return_its_booster_predicts(served: Path) -> None:
    row = freeze("C2", served)
    _predicted(SCORED)

    score_candidate("C2", SCORED[-1], panel_calendar())

    model = load_artifact(row)
    rows = day_rows(SCORED[1], served, row.recipe["features"])
    shadow = ModelRun(
        run_id="c2",
        model_type="lightgbm",
        objective="regression",
        feature_list=row.recipe["features"],
    )
    direct = model["booster"].predict(model_inputs(shadow, model, rows))
    expected = pd.Series(np.asarray(direct), index=rows["stock_id"].to_numpy())
    stored = _scores("C2", SCORED[1])
    np.testing.assert_allclose(stored["raw"], expected.reindex(stored.index))
    np.testing.assert_allclose(stored["score"], stored["raw"])
    universe = stored[stored["in_universe"]]
    np.testing.assert_allclose(universe["pct"], universe["score"].rank(pct=True))
    assert ShadowScore.objects.get(candidate="C2", date=SCORED[1], stock_id="1139").pct is None
    assert set(stored["serving_run_id"]) == {"return-2016-11.lightgbm.h5.f000"}


def test_scoring_again_writes_no_second_row(served: Path) -> None:
    freeze("C1", served)
    _predicted(SCORED)
    score_candidate("C1", SCORED[-1], panel_calendar())
    count = ShadowScore.objects.count()

    again = score_candidate("C1", SCORED[-1], panel_calendar())

    assert again.scored == []
    assert ShadowScore.objects.count() == count


def test_without_an_artifact_nothing_is_scored(served: Path) -> None:
    _predicted(SCORED)

    got = score_candidate("C1", SCORED[-1], panel_calendar())

    assert (got.scored, got.notes) == ([], [])
    assert not ShadowScore.objects.exists()


def test_a_changed_file_scores_nothing_until_a_next_generation_replaces_it(
    served: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = freeze("C1", served)
    _predicted(SCORED)
    path = Path(settings.MODEL_DIR) / row.path
    path.write_bytes(path.read_bytes() + b"x")

    broken = score_candidate("C1", SCORED[-1], panel_calendar())

    assert not ShadowScore.objects.exists()
    (note,) = broken.notes
    assert (note["book"], note["signal_date"]) == ("C1", "2016-12-01")
    assert "no longer matches its frozen sha256" in note["artifact_invalid"]
    later = date(2016, 12, 5)
    monkeypatch.setattr(candidates.timezone, "localdate", lambda: later)
    second = freeze("C1", served, replace=True)
    assert second.first_signal == later
    _predicted([later])

    replaced = score_candidate("C1", later, panel_calendar())

    assert (replaced.scored, replaced.notes) == (["2016-12-05"], [])
    assert set(ShadowScore.objects.values_list("date", "generation")) == {(later, 2)}


def test_a_frozen_candidate_falls_behind_once_its_slot_serves_another_run(served: Path) -> None:
    assert unfollowed() == []
    freeze("C1", served)
    assert unfollowed() == []
    ModelRun.objects.filter(run_id=RANK_RUN).update(status=ModelStatus.RETIRED)
    _serving("rank-2016-12", ModelObjective.RANK, ["tech_ret_1d"])

    assert unfollowed() == ["C1"]
