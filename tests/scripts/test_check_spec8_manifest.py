import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_spec8_manifest import base_node_id, compare, parse_node_ids, read_manifest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_spec8_manifest.py"
PYTEST_INI = "[pytest]\nmarkers =\n    spec8: mandatory spec section 8 test\n"
TEST_MODULE = (
    "import pytest\n\n\n"
    "@pytest.mark.spec8\n"
    "def test_listed() -> None:\n"
    "    assert True\n\n\n"
    "@pytest.mark.spec8\n"
    '@pytest.mark.parametrize("n", [1, 2])\n'
    "def test_param(n: int) -> None:\n"
    "    assert n\n\n\n"
    "def test_unmarked() -> None:\n"
    "    assert True\n"
)


@pytest.fixture
def suite(tmp_path: Path) -> Path:
    (tmp_path / "pytest.ini").write_text(PYTEST_INI, encoding="utf-8")
    (tmp_path / "test_a.py").write_text(TEST_MODULE, encoding="utf-8")
    return tmp_path


def _run(suite: Path, manifest_lines: list[str]) -> subprocess.CompletedProcess[str]:
    manifest = suite / "manifest.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    pytest_args = ["-c", str(suite / "pytest.ini"), "--rootdir", str(suite), "-p", "no:django"]
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--", *pytest_args, str(suite)],
        capture_output=True,
        text=True,
        check=False,
        cwd=suite,
    )


def test_read_manifest_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "m.txt"
    path.write_text("\n tests/a.py::test_x \n\ntests/b.py::test_y\n", encoding="utf-8")
    assert read_manifest(path) == {"tests/a.py::test_x", "tests/b.py::test_y"}


def test_base_node_id_drops_parametrize_suffix() -> None:
    assert base_node_id("tests/a.py::test_x[case-1]") == "tests/a.py::test_x"
    assert base_node_id("tests/a.py::test_x") == "tests/a.py::test_x"


def test_compare_reports_both_directions() -> None:
    problems = compare({"a::listed", "a::gone"}, {"a::listed", "a::new"})
    assert problems == [
        "listed in manifest but not collected: a::gone",
        "marked spec8 but not in manifest: a::new",
    ]


def test_matching_manifest_passes(suite: Path) -> None:
    result = _run(suite, ["test_a.py::test_listed", "test_a.py::test_param"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "matches 2 collected test(s)" in result.stdout


def test_manifest_entry_without_test_fails(suite: Path) -> None:
    lines = ["test_a.py::test_listed", "test_a.py::test_param", "test_a.py::test_missing"]
    result = _run(suite, lines)
    assert result.returncode == 1
    assert result.stdout.splitlines() == [
        "listed in manifest but not collected: test_a.py::test_missing"
    ]


def test_marked_test_missing_from_manifest_fails(suite: Path) -> None:
    result = _run(suite, ["test_a.py::test_listed"])
    assert result.returncode == 1
    assert result.stdout.splitlines() == ["marked spec8 but not in manifest: test_a.py::test_param"]


def test_empty_manifest_with_no_marked_tests_passes(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text(PYTEST_INI, encoding="utf-8")
    (tmp_path / "test_b.py").write_text("def test_plain() -> None:\n    assert True\n")
    result = _run(tmp_path, [])
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_manifest_file_exits_2(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(tmp_path / "absent.txt")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "could not run" in result.stderr


def test_collection_error_exits_2(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text(PYTEST_INI, encoding="utf-8")
    (tmp_path / "test_broken.py").write_text("import not_a_real_module\n", encoding="utf-8")
    result = _run(tmp_path, [])
    assert result.returncode == 2
    assert "could not run" in result.stderr


def test_parse_node_ids_ignores_warning_summary_lines() -> None:
    output = (
        "test_a.py::test_listed\n"
        "test_a.py::test_param[1]\n"
        "\n"
        "=============== warnings summary ===============\n"
        "test_a.py:5: UserWarning: module level :: warning\n"
        "  warnings.warn('module level :: warning')\n"
    )
    assert parse_node_ids(output) == {"test_a.py::test_listed", "test_a.py::test_param"}


def test_collection_warnings_with_double_colon_do_not_fail(suite: Path) -> None:
    module = (suite / "test_a.py").read_text(encoding="utf-8")
    warned = "import warnings\n\nwarnings.warn('module level :: warning')\n" + module
    (suite / "test_a.py").write_text(warned, encoding="utf-8")
    result = _run(suite, ["test_a.py::test_listed", "test_a.py::test_param"])
    assert result.returncode == 0, result.stdout + result.stderr
