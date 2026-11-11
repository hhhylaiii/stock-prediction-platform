"""Holdout guard: a scope runs its holdout folds once unless forced with a reason (SPEC §7)."""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import IntegrityError, transaction
from django.db.models import Max
from django.utils import timezone

from apps.research.models import (
    HoldoutRelease,
    HoldoutRun,
    HoldoutScope,
    HoldoutVerdict,
    ModelRun,
)
from apps.research.walkforward import Fold, WalkForwardConfig

KEY_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}", re.ASCII)
SPEC_HOLDOUT_START = WalkForwardConfig().holdout_start
PRODUCTION_FOLD = 0
PRODUCTION = "production"


class HoldoutError(RuntimeError):
    """A holdout fold was about to run without a recorded claim, or a claim was refused."""


@dataclass(frozen=True)
class HoldoutTicket:
    scope: str
    run_key: str
    attempt: int
    last_date: date


@dataclass(frozen=True)
class ProductionPass:
    """Leave for production runs to write past the holdout start once a claim froze the design."""

    scope: str
    frozen_by: str


def require_key(value: str) -> str:
    """Return value if it is a plain name that is safe as one path segment."""
    if not KEY_PATTERN.fullmatch(value):
        raise ValueError(
            f"{value!r} must be 1 to 64 lowercase ASCII letters, digits, '_', '.' or '-', "
            "starting with a letter or digit"
        )
    return value


def _last_attempt(scope: str) -> int:
    last = HoldoutRun.objects.filter(scope=scope).aggregate(last=Max("attempt"))["last"]
    return int(last or 0)


def claim_holdout(
    scope: str, run_key: str, last_date: date, *, force: bool = False, reason: str = ""
) -> HoldoutTicket:
    """Record a holdout run on data through last_date; a rerun needs force, a reason, a new key."""
    if scope not in HoldoutScope.values:
        raise ValueError(f"scope must be one of {HoldoutScope.values}, not {scope!r}")
    require_key(run_key)
    note = reason.strip()
    if force and not note:
        raise HoldoutError("a forced holdout rerun needs a reason")
    if HoldoutRun.objects.filter(run_key=run_key).exists():
        raise HoldoutError(f"{run_key} already holds a holdout run; a rerun needs a new run_key")
    if ModelRun.objects.filter(run_key=run_key, params__has_key=PRODUCTION).exists():
        raise HoldoutError(f"{run_key} holds a production build; the holdout needs its own run_key")
    attempt = _last_attempt(scope) + 1 if force else 1
    try:
        with transaction.atomic(durable=True):
            HoldoutRun.objects.create(
                scope=scope, attempt=attempt, run_key=run_key, last_date=last_date, reason=note
            )
    except IntegrityError as error:
        hint = "another claim took it first" if force else "rerun only with force and a reason"
        raise HoldoutError(
            f"holdout attempt {attempt} for {scope} is already recorded; {hint}"
        ) from error
    return HoldoutTicket(scope, run_key, attempt, last_date)


def holdout_claimed(scope: str) -> bool:
    """Whether scope has run its holdout at least once."""
    return HoldoutRun.objects.filter(scope=scope).exists()


def latest_ticket(scope: str) -> HoldoutTicket | None:
    """Ticket of the latest recorded claim, to resume its missing holdout folds."""
    row = HoldoutRun.objects.filter(scope=scope).order_by("-attempt").first()
    if row is None:
        return None
    return HoldoutTicket(row.scope, row.run_key, row.attempt, row.last_date)


def _not_frozen(scope: str) -> HoldoutError:
    return HoldoutError(
        f"the {scope} design is not frozen; production runs wait for its holdout claim"
    )


def production_pass(scope: str) -> ProductionPass:
    """The scope's production pass, granted only once its holdout is claimed."""
    ticket = latest_ticket(scope)
    if ticket is None:
        raise _not_frozen(scope)
    return ProductionPass(scope, ticket.run_key)


def authorize_production(run_key: str, granted: ProductionPass, fold: Fold | None = None) -> None:
    """Refuse a production write before the claim, off its scope, on a holdout key or fold."""
    if not holdout_claimed(granted.scope):
        raise _not_frozen(granted.scope)
    if not run_key.startswith(f"{granted.scope}-"):
        raise HoldoutError(
            f"a {granted.scope} production pass covers {granted.scope}- run_keys, not {run_key!r}"
        )
    if HoldoutRun.objects.filter(run_key=run_key).exists():
        raise HoldoutError(f"{run_key} is a holdout run; production runs use their own run_key")
    if fold is not None and (fold.holdout or fold.index != PRODUCTION_FOLD):
        raise HoldoutError(
            f"a production pass writes production folds only, not fold {fold.index}"
            f"{' of the holdout' if fold.holdout else ''}"
        )


def is_holdout(fold: Fold) -> bool:
    """Holdout by its flag or by the SPEC holdout start, whatever config made the fold."""
    return fold.holdout or fold.test_end >= SPEC_HOLDOUT_START


def authorize_folds(folds: Iterable[Fold], run_key: str, ticket: HoldoutTicket | None) -> None:
    """Refuse holdout folds unless ticket is the latest recorded claim of its scope for run_key."""
    holdout = [fold for fold in folds if is_holdout(fold)]
    if not holdout:
        return
    if ticket is None or ticket.run_key != run_key:
        indices = [fold.index for fold in holdout]
        raise HoldoutError(f"folds {indices} are holdout; claim_holdout for {run_key} first")
    recorded = HoldoutRun.objects.filter(
        scope=ticket.scope, attempt=ticket.attempt, run_key=run_key, last_date=ticket.last_date
    ).exists()
    if not recorded:
        raise HoldoutError(
            f"no holdout_run row for {ticket.scope} attempt {ticket.attempt} and {run_key}"
        )
    if ticket.attempt != _last_attempt(ticket.scope):
        raise HoldoutError(f"attempt {ticket.attempt} of {ticket.scope} has been superseded")
    beyond = [fold.index for fold in holdout if fold.test_end > ticket.last_date]
    if beyond:
        raise HoldoutError(f"folds {beyond} run past the claimed last_date {ticket.last_date}")


PASSED, FAILED, DEGRADED = (verdict.value for verdict in HoldoutVerdict)


@dataclass(frozen=True)
class Reported:
    """A holdout's one report: its verdict, the gates it failed and the flags it raised."""

    scope: str
    status: str
    path: str
    sha256: str
    text: str
    failed: list[str]
    flagged: list[str]


def report_marker(run_key: str) -> Path:
    """File that keeps a holdout's one report and verdict under its cache directory."""
    return Path(settings.RESEARCH_CACHE_DIR) / require_key(run_key) / "holdout.report"


def reported(run_key: str) -> Reported | None:
    """The one report and verdict a holdout was marked with, or None before it."""
    marker = report_marker(run_key)
    if not marker.exists():
        return None
    try:
        stored = json.loads(marker.read_text(encoding="utf-8"))
        return Reported(**{field.name: stored[field.name] for field in fields(Reported)})
    except (ValueError, TypeError, KeyError) as error:
        named = f"{run_key}/{marker.name}"
        raise HoldoutError(
            f"{named} is not a holdout report marker: {type(error).__name__}: {error}"
        ) from error


def _verdict(scope: str, key: str) -> Reported | None:
    """The report marked for the scope's holdout; one of another scope is none."""
    marked = reported(key)
    return marked if marked is not None and marked.scope == scope else None


def release_holdout(scope: str, status: str, reason: str) -> HoldoutRelease:
    """Record the user's decision to build on the scope's latest holdout despite its verdict."""
    note = reason.strip()
    if not note:
        raise HoldoutError("a holdout release needs the user's reason")
    if status not in (FAILED, DEGRADED):
        raise HoldoutError(f"a release names {FAILED} or {DEGRADED}, not {status!r}")
    key = production_pass(scope).frozen_by
    marked = _verdict(scope, key)
    if marked is None:
        raise HoldoutError(f"holdout {key} has no report yet; there is no verdict to release")
    if marked.status == PASSED:
        raise HoldoutError(f"holdout {key} was judged {PASSED}; it needs no release")
    if marked.status != status:
        raise HoldoutError(
            f"holdout {key} was judged {marked.status}, not {status}; release the verdict it has"
        )
    try:
        with transaction.atomic():
            return HoldoutRelease.objects.create(
                holdout=HoldoutRun.objects.get(run_key=key),
                status=status,
                failed=marked.failed,
                flagged=marked.flagged,
                report_sha256=marked.sha256,
                reason=note,
            )
    except IntegrityError as error:
        raise HoldoutError(f"holdout {key} is already released") from error


def released_verdict(granted: ProductionPass) -> dict[str, Any] | None:
    """Leave to build on the scope's holdout: None when it passed, else its recorded release."""
    key = granted.frozen_by
    marked = _verdict(granted.scope, key)
    if marked is None:
        raise HoldoutError(f"holdout {key} has no report; a build waits for its verdict")
    if marked.status == PASSED:
        return None
    release = HoldoutRelease.objects.filter(
        holdout__run_key=key, status=marked.status, report_sha256=marked.sha256
    ).first()
    report = Path(marked.path).name
    if release is None and HoldoutRelease.objects.filter(holdout__run_key=key).exists():
        raise HoldoutError(
            f"holdout {key} was released for another report or verdict than its marker holds "
            f"now ({marked.status}, report {report}); the marker changed after the release"
        )
    if release is None:
        raise HoldoutError(
            f"holdout {key} was judged {marked.status} (report {report}); the build is "
            "refused until release_holdout records the user's decision"
        )
    return {
        "status": release.status,
        "failed": release.failed,
        "flagged": release.flagged,
        "reason": release.reason,
        "released_at": timezone.localtime(release.released_at).isoformat(),
        "report": marked.path,
        "sha256": release.report_sha256,
    }
