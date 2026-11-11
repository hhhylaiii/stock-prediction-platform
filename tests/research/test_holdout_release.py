"""build_run reads the holdout's verdict; a verdict short of the criteria needs a release (§7)."""

from io import StringIO
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.models import DataQualityLog
from apps.research import production
from apps.research.holdout import (
    DEGRADED,
    FAILED,
    PASSED,
    HoldoutError,
    release_holdout,
    report_marker,
)
from apps.research.management.commands.release_holdout import Command as ReleaseCommand
from apps.research.models import HoldoutRelease, ModelRun
from apps.research.production import HOLDOUT_RELEASE, RELEASE_CHECK, build_run
from apps.research.runs import PRODUCTION
from tests.research.run_panels import claim_all, weekdays, write_holdout_report

LAST = weekdays()[-1]
REPORT = "/reports/rank-final.md"


def build(panel: Path) -> production.Built:
    return build_run("rank", "rank-prod", "rank-wf1", LAST, panel)


def production_note(run_key: str) -> dict[str, object]:
    run = ModelRun.objects.filter(run_key=run_key).order_by("run_id").first()
    assert run is not None
    note: dict[str, object] = run.params[PRODUCTION]
    return note


@pytest.mark.django_db
def test_build_run_waits_for_the_holdout_s_report(researched: Path) -> None:
    claim_all(reported=None)

    with pytest.raises(HoldoutError, match="holdout rank-final has no report"):
        build(researched)
    with pytest.raises(HoldoutError, match="has no report yet; there is no verdict to release"):
        release_holdout("rank", FAILED, "the user accepts it")

    assert not ModelRun.objects.filter(run_key="rank-prod").exists()


@pytest.mark.django_db
def test_build_run_builds_on_a_holdout_that_met_its_criteria(researched: Path) -> None:
    claim_all()

    build(researched)

    assert HOLDOUT_RELEASE not in production_note("rank-prod")
    assert not DataQualityLog.objects.filter(check_name=RELEASE_CHECK).exists()
    with pytest.raises(HoldoutError, match=f"was judged {PASSED}; it needs no release"):
        release_holdout("rank", FAILED, "the user accepts it")


@pytest.mark.django_db
@pytest.mark.parametrize("status", [FAILED, DEGRADED])
def test_a_verdict_short_of_the_criteria_builds_only_after_its_release(
    researched: Path, status: str
) -> None:
    claim_all(reported=status)
    write_holdout_report("rank", "rank-final", status, REPORT)

    with pytest.raises(HoldoutError, match=f"was judged {status} \\(report {REPORT}\\)"):
        build(researched)
    assert not ModelRun.objects.filter(run_key="rank-prod").exists()
    released = release_holdout("rank", status, "  the user accepts the 5-day MAE gap  ")

    build(researched)

    note = production_note("rank-prod")[HOLDOUT_RELEASE]
    assert note == {
        "status": status,
        "reason": "the user accepts the 5-day MAE gap",
        "released_at": released.released_at.isoformat(),
        "report": REPORT,
    }
    assert HOLDOUT_RELEASE not in production_note("rank-prod-shadow")
    alert = DataQualityLog.objects.get(check_name=RELEASE_CHECK)
    assert (alert.date, alert.severity) == (LAST, DataQualityLog.Severity.WARNING)
    assert alert.detail == {
        "scope": "rank",
        "run_key": "rank-prod",
        "holdout": "rank-final",
        **note,
    }


@pytest.mark.django_db
def test_a_release_names_the_verdict_it_releases_and_is_recorded_once(researched: Path) -> None:
    claim_all(reported=DEGRADED)

    with pytest.raises(HoldoutError, match="needs the user's reason"):
        release_holdout("rank", DEGRADED, "   ")
    with pytest.raises(HoldoutError, match=f"was judged {DEGRADED}, not {FAILED}"):
        release_holdout("rank", FAILED, "the user accepts it")
    assert not HoldoutRelease.objects.exists()
    release_holdout("rank", DEGRADED, "the user accepts it")
    with pytest.raises(HoldoutError, match="rank-final is already released"):
        release_holdout("rank", DEGRADED, "again")

    assert HoldoutRelease.objects.count() == 1


@pytest.mark.django_db
def test_a_release_of_one_verdict_does_not_cover_another(researched: Path) -> None:
    claim_all(reported=DEGRADED)
    release_holdout("rank", DEGRADED, "the user accepts it")
    report_marker("rank-final").unlink()
    write_holdout_report("rank", "rank-final", FAILED, REPORT)

    with pytest.raises(HoldoutError, match=f"was judged {FAILED}"):
        build(researched)


@pytest.mark.django_db
def test_no_release_without_a_claim(run_panel: Path) -> None:
    with pytest.raises(HoldoutError, match="rank design is not frozen"):
        release_holdout("rank", FAILED, "the user accepts it")


@pytest.mark.django_db
def test_the_release_command_records_the_user_s_decision(researched: Path) -> None:
    claim_all(reported=FAILED)
    out = StringIO()

    with pytest.raises(CommandError, match="needs the user's reason"):
        call_command("release_holdout", scope="rank", status=FAILED, reason=" ")
    call_command("release_holdout", scope="rank", status=FAILED, reason="accepted", stdout=out)

    release = HoldoutRelease.objects.get()
    assert (release.holdout.run_key, release.status, release.reason) == (
        "rank-final",
        FAILED,
        "accepted",
    )
    assert f"rank-final: released {FAILED}" in out.getvalue()
    assert "只有使用者明確同意後才能執行" in ReleaseCommand.help
