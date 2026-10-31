"""Fail when lines added since a base ref carry `#` comments other than tool directives."""

import argparse
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

ALLOWED_DIRECTIVES = (
    re.compile(r"# noqa(: ?[A-Z]+[0-9]+(, ?[A-Z]+[0-9]+)*)?"),
    re.compile(r"# type: ignore(\[[a-z0-9_-]+(, ?[a-z0-9_-]+)*\])?"),
    re.compile(r"# pragma: no (cover|branch)"),
    re.compile(r"# fmt: (off|on|skip)"),
)
ALLOWED_HELP = ", ".join(
    ("#! on line 1", "# noqa[: codes]", "# type: ignore[codes]", "# pragma: no cover", "# fmt: off")
)
CHECKED_SUFFIXES = (".py", ".toml", ".yml", ".yaml", ".sh")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
QUOTE_OPENERS = " \t=:([{,"


@dataclass(frozen=True)
class Comment:
    line: int
    column: int
    text: str


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    comment: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.comment}"


def is_checked(path: str) -> bool:
    name = Path(path).name
    return (
        name.endswith(CHECKED_SUFFIXES)
        or name == "Dockerfile"
        or name.endswith(".Dockerfile")
        or name.startswith("Dockerfile.")
    )


def is_allowed(comment: Comment) -> bool:
    """A shebang is allowed only at the very start of a file; directives must stand alone."""
    if comment.text.startswith("#!"):
        return comment.line == 1 and comment.column == 0
    text = comment.text.rstrip()
    return any(pattern.fullmatch(text) for pattern in ALLOWED_DIRECTIVES)


def added_lines(diff: str) -> set[int]:
    """Line numbers on the new side of every hunk in a single-file `--unified=0` diff."""
    lines: set[int] = set()
    for line in diff.splitlines():
        match = HUNK_RE.match(line)
        if match:
            start = int(match.group(1))
            count = int(match.group(2) or "1")
            lines.update(range(start, start + count))
    return lines


def text_comment(line: str) -> int | None:
    """Column of the first `#` that opens a comment outside quotes, if any."""
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        previous = line[index - 1] if index else " "
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote is not None:
            if char == quote:
                quote = None
        elif char in "\"'" and previous in QUOTE_OPENERS:
            quote = char
        elif char == "#" and previous.isspace():
            return index
    return None


def text_comments(source: str) -> list[Comment]:
    found: list[Comment] = []
    for number, line in enumerate(source.splitlines(), start=1):
        column = text_comment(line)
        if column is not None:
            found.append(Comment(number, column, line[column:].rstrip()))
    return found


def python_comments(source: str) -> list[Comment]:
    """Comment tokens of a Python file; falls back to text scanning on invalid source."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return text_comments(source)
    return [
        Comment(tok.start[0], tok.start[1], tok.string)
        for tok in tokens
        if tok.type == tokenize.COMMENT
    ]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "core.quotePath=false", *args],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8", errors="replace")


def changed_files(base: str, repo: Path) -> list[str]:
    output = git(repo, "diff", "--name-only", "-z", "--diff-filter=ACMR", f"{base}...HEAD")
    return sorted(path for path in output.split("\0") if path and is_checked(path))


def find_violations(base: str, repo: Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in changed_files(base, repo):
        diff = git(
            repo,
            "diff",
            "--unified=0",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            f"{base}...HEAD",
            "--",
            path,
        )
        lines = added_lines(diff)
        source = git(repo, "show", f"HEAD:{path}")
        comments = python_comments(source) if path.endswith(".py") else text_comments(source)
        violations.extend(
            Violation(path, c.line, c.text)
            for c in comments
            if c.line in lines and not is_allowed(c)
        )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="base ref to diff against, e.g. origin/develop")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        violations = find_violations(args.base, args.repo)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        print(f"git failed: {stderr}", file=sys.stderr)
        return 2
    for violation in violations:
        print(violation)
    if violations:
        print(f"{len(violations)} disallowed comment(s); only {ALLOWED_HELP}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
