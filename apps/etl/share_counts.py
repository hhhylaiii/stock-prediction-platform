"""Cleanup shared by every writer of shares_outstanding (SPEC §4)."""

from collections.abc import Iterable
from datetime import date

from django.db import connection

from apps.market.models import ShareChangeReason, SharesOutstanding

DROP_REPEATED_COUNTS_SQL = """
DELETE FROM shares_outstanding s
USING (
    SELECT stock_id, effective_date, shares,
           LAG(shares) OVER (PARTITION BY stock_id ORDER BY effective_date) AS previous
    FROM shares_outstanding
    WHERE stock_id = ANY(%(stocks)s)
) ordered
WHERE s.stock_id = ordered.stock_id
  AND s.effective_date = ordered.effective_date
  AND ordered.shares = ordered.previous
  AND s.reason = %(unknown)s
"""


def drop_repeated_counts(stock_ids: Iterable[str]) -> int:
    """Delete unknown-reason rows repeating the count of the row before; returns rows removed."""
    stocks = sorted(set(stock_ids))
    if not stocks:
        return 0
    with connection.cursor() as cursor:
        cursor.execute(
            DROP_REPEATED_COUNTS_SQL, {"stocks": stocks, "unknown": ShareChangeReason.UNKNOWN}
        )
        return int(cursor.rowcount)


def latest_counts(before: date) -> dict[str, int]:
    """The last shares_outstanding count of each stock effective before the given day."""
    latest = (
        SharesOutstanding.objects.filter(effective_date__lt=before)
        .order_by("stock_id", "-effective_date")
        .distinct("stock_id")
        .values_list("stock_id", "shares")
    )
    return dict(latest)
