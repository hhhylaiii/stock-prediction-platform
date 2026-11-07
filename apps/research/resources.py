"""Free memory of the host and the worker's own peak, for jobs too heavy to start blind."""

import re
import resource
import shutil
import subprocess
import sys
from pathlib import Path

MEMINFO = Path("/proc/meminfo")
PRESSURE = re.compile(r"free percentage:\s*(\d+)%")
PRESSURE_TIMEOUT = 10


def _meminfo(text: str) -> float | None:
    values = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        fields = rest.split()
        if fields and fields[0].isdigit():
            values[name] = int(fields[0])
    total, available = values.get("MemTotal"), values.get("MemAvailable")
    return None if not total or available is None else 100 * available / total


def _pressure() -> float | None:
    tool = shutil.which("memory_pressure")
    if tool is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            [tool, "-Q"], capture_output=True, text=True, timeout=PRESSURE_TIMEOUT
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = PRESSURE.search(out)
    return None if found is None else float(found.group(1))


def free_percent() -> float | None:
    """Available memory as a percent of total: /proc/meminfo in the worker, else memory_pressure."""
    if MEMINFO.exists():
        return _meminfo(MEMINFO.read_text())
    return _pressure()


def peak_rss_mb() -> float:
    """This process's resident high-water mark so far, in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform.startswith("linux") else peak / 1024**2
