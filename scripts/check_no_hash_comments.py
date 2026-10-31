"""Fail when lines added since a base ref carry `#` comments other than tool directives."""

import argparse
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

ALLOWED_PREFIXES = ("#!", "# noqa", "# type:", "# pragma:", "# fmt:")
CHECKED_SUFFIXES = (".py", ".toml", ".yml", ".yaml", ".sh")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    comment: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.comment}"


def is_checked(path: str) -> bool:
    name = Path(path).name
    return name.endswith(CHECKED_SUFFIXES) or name == "Dockerfile" or name.endswith(".Dockerfile")


def is_allowed(comment: str) -> bool:
    return comment.startswith(ALLOWED_PREFIXES)


def added_lines(diff: str) -> dict[str, set[int]]:
    """Map each file in `git diff --unified=0` output to the line numbers it adds."""
    result: dict[str, set[int]] = {}
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            current = target[2:] if target.startswith("b/") else None
            continue
        match = HUNK_RE.match(line)
        if match and current is not None:
            start = int(match.group(1))
            count = int(match.group(2) or "1")
            result.setdefault(current, set()).update(range(start, start + count))
    return result


def text_comment(line: str) -> str | None:
    """Return the comment opened by the first `#` that sits outside quotes after whitespace."""
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote is not None:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[index:].rstrip()
    return None


def python_comments(source: str) -> dict[int, str]:
    """Map line numbers to comment tokens; falls back to text scanning on invalid Python."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return text_comments(source)
    return {tok.start[0]: tok.string for tok in tokens if tok.type == tokenize.COMMENT}


def text_comments(source: str) -> dict[int, str]:
    found = {n: text_comment(line) for n, line in enumerate(source.splitlines(), start=1)}
    return {n: comment for n, comment in found.items() if comment is not None}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def find_violations(base: str, repo: Path) -> list[Violation]:
    diff = git(repo, "diff", "--unified=0", "--no-color", "--diff-filter=ACMR", f"{base}...HEAD")
    violations: list[Violation] = []
    for path, lines in sorted(added_lines(diff).items()):
        if not is_checked(path):
            continue
        source = git(repo, "show", f"HEAD:{path}")
        comments = python_comments(source) if path.endswith(".py") else text_comments(source)
        violations.extend(
            Violation(path, n, comments[n])
            for n in sorted(lines & comments.keys())
            if not is_allowed(comments[n])
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
        print(f"git failed: {exc.stderr.strip()}", file=sys.stderr)
        return 2
    for violation in violations:
        print(violation)
    if violations:
        allowed = ", ".join(ALLOWED_PREFIXES)
        print(
            f"{len(violations)} disallowed comment(s); only {allowed} are allowed", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
