import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_no_hash_comments import (
    Comment,
    added_lines,
    is_allowed,
    is_checked,
    python_comments,
    text_comment,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_no_hash_comments.py"
GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.test",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.test",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, env=GIT_ENV, capture_output=True)


def _write(repo: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _repo_with_change(tmp_path: Path, base: dict[str, str], head: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, base)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-gpg-sign", "-m", "base")
    _git(repo, "branch", "base")
    _write(repo, head)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-gpg-sign", "-m", "head")
    return repo


def _run(
    repo: Path, base: str = "base", extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), base, "--repo", str(repo)],
        capture_output=True,
        text=True,
        check=False,
        env={**GIT_ENV, **(extra_env or {})},
    )


@pytest.mark.parametrize(
    "comment",
    [
        Comment(1, 0, "#!/usr/bin/env python"),
        Comment(3, 9, "# noqa"),
        Comment(3, 9, "# noqa: F403"),
        Comment(3, 9, "# noqa: S603, S607"),
        Comment(3, 9, "# type: ignore"),
        Comment(3, 9, "# type: ignore[misc, import-untyped]"),
        Comment(3, 9, "# pragma: no cover"),
        Comment(3, 9, "# fmt: off"),
    ],
)
def test_tool_directives_are_allowed(comment: Comment) -> None:
    assert is_allowed(comment)


@pytest.mark.parametrize(
    "comment",
    [
        Comment(3, 0, "# todo"),
        Comment(3, 9, "#noqa"),
        Comment(3, 9, "# Type: ignore"),
        Comment(2, 0, "#!/bin/sh"),
        Comment(1, 4, "#! sneaky"),
        Comment(3, 9, "#! sneaky note"),
        Comment(3, 9, "# noqa because legacy reasons"),
        Comment(3, 9, "# type: ignore 說明"),
        Comment(3, 9, "# pragma: no cover explained"),
        Comment(3, 9, "# fmt: off for the table"),
    ],
)
def test_plain_or_annotated_comments_are_not_allowed(comment: Comment) -> None:
    assert not is_allowed(comment)


@pytest.mark.parametrize(
    ("path", "checked"),
    [
        ("a.py", True),
        ("pyproject.toml", True),
        (".github/workflows/ci.yml", True),
        ("x.yaml", True),
        ("scripts/run.sh", True),
        ("ops/Dockerfile", True),
        ("ops/web.Dockerfile", True),
        ("Dockerfile.prod", True),
        ("README.md", False),
        (".env.example", False),
        ("tests/fixtures/tdcc/sample.csv", False),
    ],
)
def test_checked_file_types(path: str, checked: bool) -> None:
    assert is_checked(path) is checked


@pytest.mark.parametrize(
    ("line", "column"),
    [
        ("# full line", 0),
        ("  key: value  # note", 14),
        ('url: "http://x/#frag"', None),
        ("url: 'a # b'", None),
        ('echo "${#items[@]}" $#', None),
        ("value: a#b", None),
        ('say "quote \\" still # inside"', None),
        ("  - name: don't cache # explain why", 22),
        ("echo 'C:\\' # note", 11),
    ],
)
def test_text_comment_detection(line: str, column: int | None) -> None:
    assert text_comment(line) == column


def test_python_comments_ignore_hashes_inside_strings() -> None:
    source = 'a = "# not a comment"\nb = 1  # real\nc = """\n# inside docstring\n"""\n'
    assert python_comments(source) == [Comment(2, 7, "# real")]


def test_python_comments_fall_back_on_invalid_source() -> None:
    assert python_comments("def broken(:\n    x = 1  # note\n") == [Comment(2, 11, "# note")]


def test_added_lines_parses_unified_zero_hunks() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "@@ -1,0 +2,2 @@\n+x\n++++ looks like a header\n@@ -5 +7 @@\n-old\n+new\n"
        "@@ -9 +8,0 @@\n-gone\n"
    )
    assert added_lines(diff) == {2, 3, 7}


def test_flags_only_added_disallowed_comments(tmp_path: Path) -> None:
    repo = _repo_with_change(
        tmp_path,
        base={"a.py": "# legacy comment\nx = 1\n"},
        head={
            "a.py": (
                "# legacy comment\nx = 1\n"
                'y = "# not a comment"\n'
                "z = 2  # noqa: E501\n"
                "w = 3  # todo\n"
            ),
            "b.sh": '#!/usr/bin/env bash\n# explain\necho "${#arr[@]}"\n',
            "c.yml": 'key: value # note\nurl: "http://x/#frag"\n',
            "ops/Dockerfile": "# base image\nFROM python:3.12-slim\n",
            "notes.md": "# Heading\n",
        },
    )
    result = _run(repo)
    assert result.returncode == 1
    assert result.stdout.splitlines() == [
        "a.py:5: # todo",
        "b.sh:2: # explain",
        "c.yml:1: # note",
        "ops/Dockerfile:1: # base image",
    ]


def test_passes_when_only_directives_added(tmp_path: Path) -> None:
    repo = _repo_with_change(
        tmp_path,
        base={"a.py": "x = 1\n"},
        head={
            "a.py": "x = 1\nfrom m import *  # noqa: F403\n",
            "run.sh": "#!/bin/sh\necho ok\n",
        },
    )
    result = _run(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""


def test_unknown_base_ref_exits_2(tmp_path: Path) -> None:
    repo = _repo_with_change(tmp_path, base={"a.py": "x = 1\n"}, head={"a.py": "x = 2\n"})
    result = _run(repo, base="no-such-ref")
    assert result.returncode == 2
    assert "git failed" in result.stderr


def test_unusual_paths_and_configs_are_still_checked(tmp_path: Path) -> None:
    repo = _repo_with_change(
        tmp_path,
        base={"keep.py": "x = 1\n"},
        head={
            "中文/模組.py": "y = 1  # nonascii\n",
            "dir with space/a.py": "z = 1  # spaced\n",
            "ops/Dockerfile.prod": "# stage note\nFROM python:3.12-slim\n",
            ".github/w.yml": "  - name: don't cache # explain why\n",
        },
    )
    noprefix = {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "diff.noprefix",
        "GIT_CONFIG_VALUE_0": "true",
    }
    result = _run(repo, extra_env=noprefix)
    assert result.returncode == 1, result.stderr
    assert result.stdout.splitlines() == [
        ".github/w.yml:1: # explain why",
        "dir with space/a.py:1: # spaced",
        "ops/Dockerfile.prod:1: # stage note",
        "中文/模組.py:1: # nonascii",
    ]


def test_non_utf8_checked_file_reports_instead_of_crashing(tmp_path: Path) -> None:
    repo = _repo_with_change(tmp_path, base={"a.py": "x = 1\n"}, head={"a.py": "x = 2\n"})
    sample = repo / "tests" / "fixtures" / "sample.yml"
    sample.parent.mkdir(parents=True)
    sample.write_bytes("name: 台積電 # 註\n".encode("big5"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-gpg-sign", "-m", "big5")
    result = _run(repo)
    assert result.returncode == 1, result.stderr
    assert len(result.stdout.splitlines()) == 1
    assert result.stdout.startswith("tests/fixtures/sample.yml:1: #")
    assert "Traceback" not in result.stderr


def test_renamed_file_only_checks_lines_added_in_the_change(tmp_path: Path) -> None:
    repo = _repo_with_change(
        tmp_path,
        base={"old.py": "# legacy\nx = 1\n"},
        head={"other.py": "z = 0\n"},
    )
    _git(repo, "mv", "old.py", "new.py")
    (repo / "new.py").write_text("# legacy\nx = 1\ny = 2  # added\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-gpg-sign", "-m", "rename")
    result = _run(repo)
    assert result.returncode == 1, result.stderr
    assert result.stdout.splitlines() == ["new.py:3: # added"]
