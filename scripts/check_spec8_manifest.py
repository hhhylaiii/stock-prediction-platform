"""Fail when tests marked `spec8` and the node ids in the spec8 manifest disagree."""

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_MANIFEST = Path("tests/spec8_manifest.txt")
NO_TESTS_COLLECTED = 5
NODE_ID_RE = re.compile(r"^[^\s:]+\.py::\S")


class CollectionError(RuntimeError):
    """Pytest could not collect the test suite."""


def read_manifest(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def base_node_id(node_id: str) -> str:
    """Drop the parametrize suffix so `test_x[a]` and `test_x[b]` count as `test_x`."""
    return node_id.split("[", 1)[0]


def collect_spec8(pytest_args: Sequence[str]) -> set[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "spec8",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        *pytest_args,
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode not in (0, NO_TESTS_COLLECTED):
        raise CollectionError(proc.stdout + proc.stderr)
    return parse_node_ids(proc.stdout)


def parse_node_ids(output: str) -> set[str]:
    """Node ids from `--collect-only -q`, which lists them before the first blank line."""
    node_ids: set[str] = set()
    for line in output.splitlines():
        if not line.strip():
            break
        if NODE_ID_RE.match(line):
            node_ids.add(base_node_id(line.strip()))
    return node_ids


def compare(manifest: set[str], collected: set[str]) -> list[str]:
    missing = [f"listed in manifest but not collected: {n}" for n in sorted(manifest - collected)]
    unlisted = [f"marked spec8 but not in manifest: {n}" for n in sorted(collected - manifest)]
    return missing + unlisted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("pytest_args", nargs="*", help="extra pytest arguments after --")
    args = parser.parse_args(argv)
    try:
        manifest = read_manifest(args.manifest)
        collected = collect_spec8(args.pytest_args)
    except (OSError, CollectionError) as exc:
        print(f"spec8 manifest check could not run: {exc}", file=sys.stderr)
        return 2
    problems = compare(manifest, collected)
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(f"spec8 manifest matches {len(manifest)} collected test(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
