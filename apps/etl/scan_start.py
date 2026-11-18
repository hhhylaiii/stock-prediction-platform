"""The first quarter and month the daily scans dated themselves (SPEC §8.5)."""

from django.conf import settings

from apps.etl.models import EtlRun
from apps.etl.revenue_scan import open_months
from apps.etl.statement_summary import open_periods

STATEMENT_SCAN = "statement_scan"
REVENUE_SCAN = "revenue_scan"


def first_scanned() -> str | None:
    """The first quarter the daily scan read as being filed; from it on, the first sight stands."""
    run = EtlRun.objects.filter(source="mops", dataset=STATEMENT_SCAN).order_by("start_date")
    first = run.first()
    if first is None:
        return None
    periods = open_periods(first.start_date, int(settings.STATEMENT_SCAN_DAYS_AFTER))
    return min(periods) if periods else None


def first_scanned_month() -> str | None:
    """The first month a successful revenue scan read as being filed."""
    runs = EtlRun.objects.filter(
        source="mops", dataset=REVENUE_SCAN, status=EtlRun.Status.SUCCESS
    ).order_by("start_date")
    first = runs.values_list("start_date", flat=True).first()
    if first is None:
        return None
    return open_months(first, int(settings.REVENUE_SCAN_DAYS_AFTER))[0]
