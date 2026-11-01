import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SPLIT = re.compile(
    r"KFold|ShuffleSplit|train_test_split|shuffle\s*=\s*True"
    r"|cross_val_(score|predict)|cross_validate|\b[A-Z]\w*CV\b"
    r"|\b(lgb|lightgbm|xgb|xgboost)\.cv\b|from\s+(lightgbm|xgboost)\s+import\s+[^\n]*\bcv\b"
    r"|LeaveOne\w*Out|LeaveP\w*Out|PredefinedSplit|learning_curve|validation_curve"
    r"|permutation_test_score|from\s+sklearn\.utils\s+import\s+[^\n]*\bshuffle\b"
)


@pytest.mark.parametrize(
    "line",
    [
        "from sklearn.model_selection import KFold",
        "cv = StratifiedKFold(5)",
        "GroupKFold(n_splits=4)",
        "ShuffleSplit(n_splits=3)",
        "x_train, x_test = train_test_split(x)",
        "DataLoader(data, shuffle=True)",
        "cross_val_score(model, x, y)",
        "cross_validate(model, x, y)",
        "model = RidgeCV(alphas=grid)",
        "LogisticRegressionCV(Cs=10)",
        "GridSearchCV(model, grid)",
        "lgb.cv(params, train_set)",
        "xgboost.cv(params, dtrain)",
        "from lightgbm import Dataset, cv",
        "LeaveOneOut()",
        "LeaveOneGroupOut()",
        "LeavePGroupsOut(2)",
        "PredefinedSplit(fold)",
        "learning_curve(model, x, y)",
        "validation_curve(model, x, y)",
        "permutation_test_score(model, x, y)",
        "from sklearn.utils import resample, shuffle",
    ],
)
def test_guard_pattern_catches_random_splits(line: str) -> None:
    assert RANDOM_SPLIT.search(line)


@pytest.mark.parametrize(
    "line",
    [
        "TimeSeriesSplit(n_splits=5)",
        "generate_folds(calendar, last_date)",
        "rng.permutation(labels)",
        '"""Never a random K-fold."""',
        "shuffle=False",
        "QLIKE",
    ],
)
def test_guard_pattern_allows_time_ordered_code(line: str) -> None:
    assert not RANDOM_SPLIT.search(line)


def test_no_random_split_in_production_code() -> None:
    """SPEC §8.11: metrics from random splits are invalid, so production code never builds one."""
    paths = [REPO_ROOT / "manage.py"] + [
        path for top in ("apps", "config", "scripts") for path in (REPO_ROOT / top).rglob("*.py")
    ]
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path in paths
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if RANDOM_SPLIT.search(line)
    ]
    assert len(paths) > 10
    assert offenders == []
