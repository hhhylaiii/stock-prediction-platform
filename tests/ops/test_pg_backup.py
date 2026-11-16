import plistlib
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import create_autospec

import pytest
from django.core.management import call_command

from apps.etl import task_alerts

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "ops" / "backup" / "pg_backup.sh"
PLIST = REPO / "ops" / "launchd" / "tw.stock-prediction-platform.pg-backup.plist"
DUMP_COMMAND = 'exec pg_dump -Fc -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
FAKE_BYTES = b"PGDMP-fake"
OLD = [f"pg-202601{day:02d}-030000.dump" for day in range(1, 9)]
FAKE_DOCKER = """#!/bin/sh
n=$(ls "$FAKE_CALLS" | wc -l | tr -d ' ')
for arg in "$@"; do printf '%s\\0' "$arg"; done > "$FAKE_CALLS/$((n + 1))"
pwd -P > "$FAKE_CWDS/$((n + 1))"
case " $* " in
  *" worker "*)
    if [ -n "${FAKE_ALERT_SLEEP:-}" ]; then sleep "$FAKE_ALERT_SLEEP"; fi
    exit "${FAKE_ALERT_STATUS:-0}" ;;
  *" db "*)
    if [ -n "${FAKE_DUMP_SLEEP:-}" ]; then sleep "$FAKE_DUMP_SLEEP"; fi
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
        self.cwds = tmp_path / "cwds"
        self.bin = tmp_path / "bin"
        for folder in (self.home, self.backups, self.calls, self.cwds, self.bin):
            folder.mkdir()
        docker = self.bin / "docker"
        docker.write_text(FAKE_DOCKER, encoding="utf-8")
        docker.chmod(0o755)

    def env(self, **extra: str) -> dict[str, str]:
        base = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.home),
            "PG_BACKUP_DIR": str(self.backups),
            "FAKE_CALLS": str(self.calls),
            "FAKE_CWDS": str(self.cwds),
        }
        return {**base, **extra}

    def __call__(self, cwd: Path = REPO, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/sh", str(SCRIPT)],
            cwd=cwd,
            env=self.env(**env),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def cwd_of(self, call: int) -> str:
        return (self.cwds / str(call)).read_text(encoding="utf-8").strip()

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
    [alert] = run.alert_calls()
    assert _context(alert).startswith("step=keep_count status=1 seconds=")


def test_the_alert_calls_the_workers_task_failure_report(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(FAKE_DUMP_STATUS="2")
    [alert] = run.alert_calls()
    assert alert[:4] == ["compose", "exec", "-T", "-e"]
    assert alert[5:10] == ["worker", "python", "manage.py", "shell", "-c"]
    report = create_autospec(task_alerts.report_failure)
    monkeypatch.setattr("apps.etl.task_alerts.report_failure", report)
    monkeypatch.setenv("PG_BACKUP_CONTEXT", _context(alert))

    call_command("shell", command=alert[10])

    report.assert_called_once()
    name, exc, context = report.call_args.args
    assert name == "ops.pg_backup"
    assert type(exc).__name__ == "PgBackupFailed"
    assert context == _context(alert)
    assert context.startswith("step=pg_dump status=2 seconds=")


def test_a_folder_that_cannot_be_made_still_alerts_from_the_repo(run: Run, tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder", encoding="utf-8")

    result = run(cwd=Path("/"), PG_BACKUP_DIR=str(blocker / "backups"))

    assert result.returncode != 0
    [alert] = run.alert_calls()
    assert _context(alert).startswith("step=folder status=1 seconds=")
    assert run.cwd_of(len(run.docker_calls())) == str(REPO)
    assert all("db" not in call for call in run.docker_calls())


def test_a_stopped_run_removes_its_partial_file_and_alerts(run: Run) -> None:
    process = subprocess.Popen(
        ["/bin/sh", str(SCRIPT)],
        cwd=REPO,
        env=run.env(FAKE_DUMP_SLEEP="2"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while not (run.calls / "1").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert run.leftovers(), "the dump never started"

    process.send_signal(signal.SIGTERM)

    assert process.wait(timeout=30) != 0
    assert run.leftovers() == []
    assert run.dumps() == []
    assert "failed step=signal status=1 seconds=" in run.log()
    [alert] = run.alert_calls()
    assert _context(alert).startswith("step=signal status=1 seconds=")


def test_a_stop_while_alerting_does_not_fail_twice(run: Run) -> None:
    process = subprocess.Popen(
        ["/bin/sh", str(SCRIPT)],
        cwd=REPO,
        env=run.env(FAKE_DUMP_STATUS="1", FAKE_ALERT_SLEEP="2"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while not (run.calls / "2").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert run.alert_calls(), "the alert never started"

    process.send_signal(signal.SIGTERM)

    assert process.wait(timeout=30) != 0
    assert run.log().count(" failed ") == 1
    assert len(run.alert_calls()) == 1
    assert "alert handed to the worker" in run.log()


def test_the_launchd_job_runs_the_script_at_three_with_docker_on_its_path() -> None:
    job = plistlib.loads(PLIST.read_bytes())

    assert job["Label"] == PLIST.stem
    assert job["StartCalendarInterval"] == {"Hour": 3, "Minute": 0}
    assert job["EnvironmentVariables"]["PATH"].split(":")[0] == "/usr/local/bin"
    shell, flag, command = job["ProgramArguments"]
    assert (shell, flag) == ("/bin/sh", "-c")
    checkout = "$HOME/Project/stock-prediction-platform"
    assert command == f'exec /bin/sh "{checkout}/{SCRIPT.relative_to(REPO)}"'
