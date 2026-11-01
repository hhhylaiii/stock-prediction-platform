import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SPLIT = re.compile(
    r"KFold|ShuffleSplit|TimeSeriesSplit|train_test_split|shuffle\s*=\s*True"
    r"|cross_val_(score|predict)|cross_validate|\b[A-Z]\w*CV\b"
    r"|\b(lgbm?|lightgbm|xgb|xgboost)\.cv\b|from\s+(lightgbm|xgboost)\s+import\s+[^\n]*\bcv\b"
    r"|LeaveOne\w*Out|LeaveP\w*Out|PredefinedSplit|learning_curve|validation_curve"
    r"|permutation_test_score|from\s+sklearn\.utils\s+import\s+[^\n]*\bshuffle\b"
    r"|\.sample\(\s*frac|\brandom_split\b|validation_fraction|early_stopping\s*=\s*True"
    r"|HistGradientBoosting"
)
BANNED_NAMES = re.compile(
    r"^(sklearn\.model_selection(\..+)?|sklearn\.utils\.shuffle|(lgbm?|lightgbm|xgb|xgboost)\.cv)$"
)


def dotted(node: ast.expr) -> str:
    """Dotted name of an attribute chain such as sklearn.utils.shuffle; empty for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def banned_lines(source: str) -> list[int]:
    """Lines that build a random split, by pattern or by imported or dotted name."""
    lines = {n for n, line in enumerate(source.splitlines(), start=1) if RANDOM_SPLIT.search(line)}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [f"{node.module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.Attribute):
            names = [dotted(node)]
        else:
            continue
        if any(BANNED_NAMES.match(name) for name in names):
            lines.add(node.lineno)
    return sorted(lines)


@pytest.mark.parametrize(
    "source",
    [
        "from sklearn.model_selection import KFold",
        "cv = StratifiedKFold(5)",
        "GroupKFold(n_splits=4)",
        "ShuffleSplit(n_splits=3)",
        "TimeSeriesSplit(n_splits=5)",
        "x_train, x_test = train_test_split(x)",
        "DataLoader(data, shuffle=True)",
        "cross_val_score(model, x, y)",
        "cross_validate(model, x, y)",
        "model = RidgeCV(alphas=grid)",
        "LogisticRegressionCV(Cs=10)",
        "GridSearchCV(model, grid)",
        "lgb.cv(params, train_set)",
        "lgbm.cv(params, train_set)",
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
        "train = panel.sample(frac=0.8)",
        "random_split(data, [0.8, 0.2])",
        "model = Model(validation_fraction=0.1)",
        "model = Model(early_stopping=True)",
        "HistGradientBoostingRegressor()",
        "from sklearn.utils import (\n    resample,\n    shuffle,\n)",
        "from lightgbm import (\n    Dataset,\n    cv,\n)",
        "import sklearn.utils\nx = sklearn.utils.shuffle(x)",
        "from sklearn import model_selection",
        "import sklearn.model_selection",
    ],
)
def test_guard_catches_random_splits(source: str) -> None:
    assert banned_lines(source)


@pytest.mark.parametrize(
    "source",
    [
        "generate_folds(calendar, last_date)",
        "rng.permutation(labels)",
        '"""Never a random K-fold."""',
        "shuffle=False",
        "QLIKE",
        "rng.sample(population, 20)",
        "callbacks = [lgb.early_stopping(100)]",
        "from sklearn.linear_model import Ridge",
    ],
)
def test_guard_allows_time_ordered_code(source: str) -> None:
    assert banned_lines(source) == []


def test_no_random_split_in_production_code() -> None:
    """SPEC §8.11: metrics from random splits are invalid, so production code never builds one."""
    paths = [REPO_ROOT / "manage.py"] + [
        path for top in ("apps", "config", "scripts") for path in (REPO_ROOT / top).rglob("*.py")
    ]
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path in paths
        for lineno in banned_lines(path.read_text(encoding="utf-8"))
    ]
    assert len(paths) > 10
    assert offenders == []
