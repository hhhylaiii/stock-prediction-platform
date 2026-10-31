import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTEST_INI = "[pytest]\nmarkers =\n    spec8: mandatory spec section 8 test\n"
CONFTEST = (
    "from tests import spec8_guard\n\n"
    "pytest_runtest_makereport = spec8_guard.pytest_runtest_makereport\n"
)


def _run_suite(tmp_path: Path, module: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "pytest.ini").write_text(PYTEST_INI, encoding="utf-8")
    (tmp_path / "conftest.py").write_text(CONFTEST, encoding="utf-8")
    (tmp_path / "test_suite.py").write_text(module, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:django", "-p", "no:cacheprovider", "-q"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "decorator",
    [
        '@pytest.mark.skip(reason="no data")',
        '@pytest.mark.skipif(True, reason="no data")',
        '@pytest.mark.xfail(reason="flaky")',
    ],
)
def test_skipped_or_xfailed_spec8_test_fails_the_run(tmp_path: Path, decorator: str) -> None:
    module = (
        f"import pytest\n\n\n@pytest.mark.spec8\n{decorator}\n"
        "def test_rule() -> None:\n    assert False\n"
    )
    result = _run_suite(tmp_path, module)
    assert result.returncode == 1, result.stdout
    assert "must run" in result.stdout


def test_runtime_skip_inside_spec8_test_fails_the_run(tmp_path: Path) -> None:
    module = (
        "import pytest\n\n\n@pytest.mark.spec8\n"
        "def test_rule() -> None:\n    pytest.skip('no data')\n"
    )
    result = _run_suite(tmp_path, module)
    assert result.returncode == 1, result.stdout


def test_passing_spec8_and_skipped_ordinary_tests_pass(tmp_path: Path) -> None:
    module = (
        "import pytest\n\n\n@pytest.mark.spec8\n"
        "def test_rule() -> None:\n    assert True\n\n\n"
        '@pytest.mark.skip(reason="not spec8")\n'
        "def test_other() -> None:\n    assert False\n"
    )
    result = _run_suite(tmp_path, module)
    assert result.returncode == 0, result.stdout
    assert "1 passed, 1 skipped" in result.stdout
