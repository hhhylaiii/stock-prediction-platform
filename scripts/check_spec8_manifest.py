"""Fail when tests marked `spec8` and the node ids in the spec8 manifest disagree."""

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_MANIFEST = Path("tests/spec8_manifest.txt")
NO_TESTS_COLLECTED = 5


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
    return {base_node_id(line.strip()) for line in proc.stdout.splitlines() if "::" in line}


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
