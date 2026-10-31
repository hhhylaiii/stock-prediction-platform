import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_no_hash_comments import (
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


def _run(repo: Path, base: str = "base") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), base, "--repo", str(repo)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "comment",
    [
        "#!/usr/bin/env python",
        "# noqa: F403",
        "# type: ignore[misc]",
        "# pragma: no cover",
        "# fmt: off",
    ],
)
def test_tool_directives_are_allowed(comment: str) -> None:
    assert is_allowed(comment)


@pytest.mark.parametrize("comment", ["# todo", "#noqa", "# explain why", "# Type: ignore"])
def test_plain_comments_are_not_allowed(comment: str) -> None:
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
        ("README.md", False),
        (".env.example", False),
    ],
)
def test_checked_file_types(path: str, checked: bool) -> None:
    assert is_checked(path) is checked


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("# full line", "# full line"),
        ("  key: value  # note", "# note"),
        ('url: "http://x/#frag"', None),
        ("url: 'a # b'", None),
        ('echo "${#items[@]}" $#', None),
        ("value: a#b", None),
        ('say "quote \\" still # inside"', None),
    ],
)
def test_text_comment_detection(line: str, expected: str | None) -> None:
    assert text_comment(line) == expected


def test_python_comments_ignore_hashes_inside_strings() -> None:
    source = 'a = "# not a comment"\nb = 1  # real\nc = """\n# inside docstring\n"""\n'
    assert python_comments(source) == {2: "# real"}


def test_python_comments_fall_back_on_invalid_source() -> None:
    assert python_comments("def broken(:\n    x = 1  # note\n") == {2: "# note"}


def test_added_lines_parses_unified_zero_hunks() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "@@ -1,0 +2,2 @@\n+x\n+y\n@@ -5 +7 @@\n-old\n+new\n"
        "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-z\n"
    )
    assert added_lines(diff) == {"a.py": {2, 3, 7}}


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
