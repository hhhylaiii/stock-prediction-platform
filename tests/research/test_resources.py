import subprocess
from pathlib import Path
from typing import Any

import pytest

from apps.research import resources

MEMINFO = "MemTotal:       16000000 kB\nMemFree:         2000000 kB\nMemAvailable:    6000000 kB\n"


def test_the_worker_reads_its_free_memory_from_proc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(MEMINFO)
    monkeypatch.setattr(resources, "MEMINFO", meminfo)

    assert resources.free_percent() == pytest.approx(37.5)


def test_a_mac_reads_its_free_memory_from_memory_pressure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "MEMINFO", tmp_path / "absent")
    monkeypatch.setattr(resources.shutil, "which", lambda name: f"/usr/bin/{name}")
    asked: list[list[str]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        asked.append(args)
        return subprocess.CompletedProcess(args, 0, "System-wide memory free percentage: 47%\n")

    monkeypatch.setattr(resources.subprocess, "run", run)

    assert resources.free_percent() == 47.0
    assert asked == [["/usr/bin/memory_pressure", "-Q"]]


@pytest.mark.parametrize("text", ["MemTotal: 0 kB\nMemAvailable: 5 kB\n", "MemTotal: 10 kB\n"])
def test_an_unreadable_meminfo_gives_no_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(text)
    monkeypatch.setattr(resources, "MEMINFO", meminfo)

    assert resources.free_percent() is None


def test_no_memory_pressure_gives_no_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "MEMINFO", tmp_path / "absent")
    monkeypatch.setattr(resources.shutil, "which", lambda name: None)

    assert resources.free_percent() is None


def test_a_hung_memory_pressure_gives_no_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "MEMINFO", tmp_path / "absent")
    monkeypatch.setattr(resources.shutil, "which", lambda name: f"/usr/bin/{name}")

    def hung(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(resources.subprocess, "run", hung)

    assert resources.free_percent() is None


def test_the_peak_is_this_process_s_resident_high_water_mark_in_mb() -> None:
    assert 1 < resources.peak_rss_mb() < 1024 * 1024
