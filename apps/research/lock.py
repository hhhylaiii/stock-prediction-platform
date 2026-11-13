"""One Postgres advisory lock for the jobs that change what is served (SPEC §5 研究排程)."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import ParamSpec, TypeVar

from django.db import DatabaseError, connection
from django.utils import timezone

logger = logging.getLogger(__name__)
LOCK_ID = 74_600_001
LABEL = "research {since} {holder}"
JOBS = (
    "feature and label builds, coverage, predict, monitor, promote, restore and the scheduled "
    "research steps run one at a time"
)
LOCK_SQL = "SELECT pg_try_advisory_lock(%s), current_setting('application_name')"
LABEL_SQL = "SELECT set_config('application_name', %s, false)"
UNLOCK_SQL = "SELECT pg_advisory_unlock(%s), set_config('application_name', %s, false)"
HOLDER_SQL = """
SELECT a.pid, a.application_name
FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid
WHERE l.locktype = 'advisory' AND l.objid = %s AND l.granted
    AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database())
"""
_depth: ContextVar[int] = ContextVar("research_lock_depth", default=0)
P = ParamSpec("P")
R = TypeVar("R")


class ResearchBusyError(RuntimeError):
    """Another session holds the research lock."""


def _holder() -> str:
    with connection.cursor() as cursor:
        cursor.execute(HOLDER_SQL, [LOCK_ID])
        found = cursor.fetchone()
    return "another session" if found is None else f"'{found[1]}' (pid {found[0]})"


def _release(previous: str) -> None:
    """Unlock and restore the label; a failed unlock closes the connection, which frees it."""
    try:
        with connection.cursor() as cursor:
            cursor.execute(UNLOCK_SQL, [LOCK_ID, previous])
    except DatabaseError:
        logger.exception("research lock release failed; closing the connection to free it")
        connection.close()


@contextmanager
def research_lock(holder: str) -> Iterator[None]:
    """Hold the research lock for the block, or raise ResearchBusyError naming who holds it."""
    depth = _depth.get()
    if depth:
        token = _depth.set(depth + 1)
        try:
            yield
        finally:
            _depth.reset(token)
        return
    with connection.cursor() as cursor:
        cursor.execute(LOCK_SQL, [LOCK_ID])
        granted, previous = cursor.fetchone()
    if not granted:
        raise ResearchBusyError(f"the research lock is held by {_holder()}; {JOBS}")
    token = _depth.set(1)
    try:
        since = timezone.localtime().strftime("%m-%d %H:%M")
        with connection.cursor() as cursor:
            cursor.execute(LABEL_SQL, [LABEL.format(holder=holder, since=since)])
        yield
    finally:
        _depth.reset(token)
        _release(previous)


def locked(job: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Run the function under the research lock, labelled with the job and its first argument."""

    def wrap(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            with research_lock(f"{job} {args[0]}" if args else job):
                return func(*args, **kwargs)

        return inner

    return wrap
