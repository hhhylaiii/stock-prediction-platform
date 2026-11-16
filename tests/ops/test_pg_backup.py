import plistlib
import subprocess
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "ops" / "backup" / "pg_backup.sh"
PLIST = REPO / "ops" / "launchd" / "tw.stock-prediction-platform.pg-backup.plist"
DUMP_COMMAND = 'exec pg_dump -Fc -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
FAKE_BYTES = b"PGDMP-fake"
OLD = [f"pg-202601{day:02d}-030000.dump" for day in range(1, 9)]
FAKE_DOCKER = """#!/bin/sh
n=$(ls "$FAKE_CALLS" | wc -l | tr -d ' ')
for arg in "$@"; do printf '%s\\0' "$arg"; done > "$FAKE_CALLS/$((n + 1))"
case " $* " in
  *" worker "*) exit "${FAKE_ALERT_STATUS:-0}" ;;
  *" db "*)
    if [ "${FAKE_DUMP_EMPTY:-}" != 1 ]; then printf 'PGDMP-fake'; fi
    exit "${FAKE_DUMP_STATUS:-0}" ;;
esac
exit 99
"""


class Run:
    def __init__(self, tmp_path: Path) -> None:
        self.home = tmp_path / "home"
        self.backups = tmp_path / "backups"
        self.calls = tmp_path / "calls"
        self.bin = tmp_path / "bin"
        for folder in (self.home, self.backups, self.calls, self.bin):
            folder.mkdir()
        docker = self.bin / "docker"
        docker.write_text(FAKE_DOCKER, encoding="utf-8")
        docker.chmod(0o755)

    def __call__(self, **env: str) -> subprocess.CompletedProcess[str]:
        base = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.home),
            "PG_BACKUP_DIR": str(self.backups),
            "FAKE_CALLS": str(self.calls),
        }
        return subprocess.run(
            ["/bin/sh", str(SCRIPT)],
            env={**base, **env},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def docker_calls(self) -> list[list[str]]:
        files = sorted(self.calls.iterdir(), key=lambda path: int(path.name))
        return [path.read_text(encoding="utf-8").split("\0")[:-1] for path in files]

    def alert_calls(self) -> list[list[str]]:
        return [call for call in self.docker_calls() if "worker" in call]

    def dumps(self, folder: Path | None = None) -> list[str]:
        return sorted(path.name for path in (folder or self.backups).glob("pg-*.dump"))

    def log(self, folder: Path | None = None) -> str:
        return ((folder or self.backups) / "pg_backup.log").read_text(encoding="utf-8")

    def leftovers(self, folder: Path | None = None) -> list[Path]:
        return list((folder or self.backups).glob(".*partial"))


@pytest.fixture
def run(tmp_path: Path) -> Run:
    return Run(tmp_path)


def _context(alert: list[str]) -> str:
    [setting] = [arg for arg in alert if arg.startswith("PG_BACKUP_CONTEXT=")]
    return setting.removeprefix("PG_BACKUP_CONTEXT=")


def _old_files(run: Run) -> None:
    for name in [*OLD, "pg-manual.dump", "notes.txt"]:
        (run.backups / name).write_bytes(b"old")


def test_success_renames_the_finished_dump_into_the_default_folder(run: Run) -> None:
    result = run(PG_BACKUP_DIR="")

    folder = run.home / "Backups" / "stock-prediction-platform"
    assert result.returncode == 0, result.stdout + result.stderr
    names = run.dumps(folder)
    assert len(names) == 1
    assert (folder / names[0]).read_bytes() == FAKE_BYTES
    assert run.leftovers(folder) == []
    assert run.docker_calls() == [["compose", "exec", "-T", "db", "sh", "-c", DUMP_COMMAND]]
    log = run.log(folder)
    assert "start" in log
    assert f"end ok file={names[0]} bytes={len(FAKE_BYTES)} seconds=" in log


def test_success_keeps_only_the_newest_seven_dumps(run: Run) -> None:
    _old_files(run)

    result = run()

    assert result.returncode == 0, result.stdout + result.stderr
    names = run.dumps()
    kept_old = [name for name in names if name in OLD]
    assert kept_old == OLD[2:]
    assert len([name for name in names if name not in OLD and name != "pg-manual.dump"]) == 1
    assert (run.backups / "pg-manual.dump").exists()
    assert (run.backups / "notes.txt").exists()
    assert f"removed {OLD[0]}" in run.log()
    assert f"removed {OLD[1]}" in run.log()


def test_a_failed_dump_keeps_every_old_file_and_exits_non_zero(run: Run) -> None:
    _old_files(run)

    result = run(FAKE_DUMP_STATUS="1")

    assert result.returncode != 0
    assert [name for name in run.dumps() if name != "pg-manual.dump"] == OLD
    assert run.leftovers() == []
    assert "failed step=pg_dump status=1 seconds=" in run.log()
    [alert] = run.alert_calls()
    assert _context(alert).startswith("step=pg_dump status=1 seconds=")


def test_an_empty_dump_is_a_failure_that_keeps_every_old_file(run: Run) -> None:
    _old_files(run)

    result = run(FAKE_DUMP_EMPTY="1")

    assert result.returncode != 0
    assert [name for name in run.dumps() if name != "pg-manual.dump"] == OLD
    assert run.leftovers() == []
    assert "failed step=empty_dump status=0 seconds=" in run.log()
    [alert] = run.alert_calls()
    assert _context(alert).startswith("step=empty_dump status=0 seconds=")


def test_a_failure_whose_alert_cannot_go_out_still_exits_non_zero(run: Run) -> None:
    result = run(FAKE_DUMP_STATUS="1", FAKE_ALERT_STATUS="1")

    assert result.returncode != 0
    assert "alert not sent" in run.log()


@pytest.mark.parametrize("keep", ["0", "seven", ""])
def test_an_unreadable_keep_count_stops_before_dumping(run: Run, keep: str) -> None:
    result = run(PG_BACKUP_KEEP=keep)

    assert result.returncode != 0
    assert all("db" not in call for call in run.docker_calls())
    assert run.dumps() == []


def test_the_alert_calls_the_workers_task_failure_report(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(FAKE_DUMP_STATUS="2")
    [alert] = run.alert_calls()
    assert alert[:4] == ["compose", "exec", "-T", "-e"]
    assert alert[5:10] == ["worker", "python", "manage.py", "shell", "-c"]
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr("apps.etl.task_alerts.report_failure", lambda *args: calls.append(args))
    monkeypatch.setenv("PG_BACKUP_CONTEXT", _context(alert))

    call_command("shell", command=alert[10])

    [(name, exc, context)] = calls
    assert name == "ops.pg_backup"
    assert type(exc).__name__ == "PgBackupFailed"
    assert context == _context(alert)
    assert context.startswith("step=pg_dump status=2 seconds=")


def test_the_launchd_job_runs_the_script_at_three_with_docker_on_its_path() -> None:
    job = plistlib.loads(PLIST.read_bytes())

    assert job["Label"] == PLIST.stem
    assert job["StartCalendarInterval"] == {"Hour": 3, "Minute": 0}
    assert job["EnvironmentVariables"]["PATH"].split(":")[0] == "/usr/local/bin"
    shell, flag, command = job["ProgramArguments"]
    assert (shell, flag) == ("/bin/sh", "-c")
    checkout = "$HOME/Project/stock-prediction-platform"
    assert command == f'exec /bin/sh "{checkout}/{SCRIPT.relative_to(REPO)}"'
