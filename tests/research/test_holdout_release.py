"""build_run reads the holdout's verdict; a verdict short of the criteria needs a release (§7)."""

import json
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.utils import timezone

from apps.etl.models import DataQualityLog
from apps.research import production
from apps.research.holdout import (
    DEGRADED,
    FAILED,
    PASSED,
    HoldoutError,
    claim_holdout,
    release_holdout,
    report_marker,
    reported,
)
from apps.research.management.commands.release_holdout import Command as ReleaseCommand
from apps.research.models import HoldoutRelease, ModelRun
from apps.research.production import HOLDOUT_RELEASE, RELEASE_CHECK, build_run
from apps.research.runs import PRODUCTION
from tests.research.run_panels import claim_all, finish_holdout, weekdays, write_holdout_report

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
    researched: Path, status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[list[DataQualityLog]] = []
    monkeypatch.setattr(production, "dispatch", lambda rows: sent.append(list(rows)))
    claim_all(reported=status)
    write_holdout_report("rank", "rank-final", status, REPORT)

    with pytest.raises(HoldoutError, match=f"was judged {status} \\(report {REPORT}\\)"):
        build(researched)
    assert not ModelRun.objects.filter(run_key="rank-prod").exists()
    released = release_holdout("rank", status, "  the user accepts the 5-day MAE gap  ")

    build(researched)

    note = production_note("rank-prod")[HOLDOUT_RELEASE]
    marked = reported("rank-final")
    assert marked is not None
    assert note == {
        "status": status,
        "failed": marked.failed,
        "flagged": marked.flagged,
        "reason": "the user accepts the 5-day MAE gap",
        "released_at": timezone.localtime(released.released_at).isoformat(),
        "report": REPORT,
        "sha256": marked.sha256,
    }
    assert (released.failed, released.flagged, released.report_sha256) == (
        marked.failed,
        marked.flagged,
        marked.sha256,
    )
    assert marked.flagged == ["MAE 差"]
    assert str(note["released_at"]).endswith("+08:00")
    assert HOLDOUT_RELEASE not in production_note("rank-prod-shadow")
    alert = DataQualityLog.objects.get(check_name=RELEASE_CHECK)
    assert (alert.date, alert.severity) == (LAST, DataQualityLog.Severity.WARNING)
    assert alert.detail == {
        "scope": "rank",
        "run_key": "rank-prod",
        "holdout": "rank-final",
        **note,
    }
    assert sent == [[alert]]


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
def test_a_release_covers_only_the_claim_it_was_recorded_for(researched: Path) -> None:
    claim_all(reported=FAILED)
    release_holdout("rank", FAILED, "the user accepts it")
    claim_holdout("rank", "rank-again", LAST, force=True, reason="labels rebuilt")
    finish_holdout("rank", "rank-again", FAILED)

    with pytest.raises(HoldoutError, match="holdout rank-again was judged"):
        build(researched)
    report_marker("rank-again").unlink()
    with pytest.raises(HoldoutError, match="holdout rank-again has no report"):
        build(researched)
    with pytest.raises(HoldoutError, match="rank-again has no report yet"):
        release_holdout("rank", FAILED, "again")

    assert not ModelRun.objects.filter(run_key="rank-prod").exists()


@pytest.mark.django_db
def test_a_release_of_one_scope_leaves_the_others_refused(researched: Path) -> None:
    claim_all(reported=FAILED)
    release_holdout("rank", FAILED, "the user accepts it")

    with pytest.raises(HoldoutError, match="holdout interval-final was judged"):
        build_run("interval", "interval-prod", "interval-wf1", LAST, researched)


@pytest.mark.django_db
def test_a_marker_of_another_scope_neither_builds_nor_releases(researched: Path) -> None:
    claim_all(reported=FAILED)
    write_holdout_report("return", "rank-final", FAILED)

    with pytest.raises(HoldoutError, match="holdout rank-final has no report"):
        build(researched)
    with pytest.raises(HoldoutError, match="rank-final has no report yet"):
        release_holdout("rank", FAILED, "the user accepts it")

    assert not HoldoutRelease.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize("stored", [json.dumps({"scope": "rank", "status": FAILED}), "{not json"])
def test_a_marker_that_cannot_be_read_is_named_in_the_refusal(
    researched: Path, stored: str
) -> None:
    claim_all(reported=FAILED)
    report_marker("rank-final").write_text(stored, encoding="utf-8")
    named = "rank-final/holdout.report is not a holdout report marker"

    with pytest.raises(HoldoutError, match=named):
        build(researched)
    with pytest.raises(CommandError, match=named):
        call_command("release_holdout", scope="rank", status=FAILED, reason="accepted")


@pytest.mark.django_db
def test_a_release_names_one_of_the_two_short_verdicts(researched: Path) -> None:
    claim_all(reported="通過")

    with pytest.raises(HoldoutError, match="holdout rank-final was judged 通過"):
        build(researched)
    with pytest.raises(HoldoutError, match=f"a release names {FAILED} or {DEGRADED}"):
        release_holdout("rank", "通過", "the user accepts it")

    assert not HoldoutRelease.objects.exists()


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
    assert "未達的門檻 ['5 日 MAE']，退化旗標 ['MAE 差']" in out.getvalue()
    assert "只有使用者明確同意後才能執行" in ReleaseCommand.help
