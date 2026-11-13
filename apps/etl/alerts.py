"""Alerts for data-quality rows: always logged, posted when a webhook is set."""

import logging
from collections import Counter
from collections.abc import Sequence
from typing import Any

import requests
from django.conf import settings

from apps.etl.models import DataQualityLog
from apps.etl.quality import alerting

logger = logging.getLogger(__name__)


LEVELS = {
    DataQualityLog.Severity.ERROR: logging.ERROR,
    DataQualityLog.Severity.WARNING: logging.WARNING,
}
HEADLINE_KINDS = 5


def dispatch(results: Sequence[DataQualityLog]) -> None:
    """Log each warning or error; a failing webhook is logged, never raised."""
    announce(alerting(results))


def announce(rows: Sequence[DataQualityLog]) -> None:
    """Log and post the rows whatever their severity, for news that is not a warning."""
    alerts = [_payload(row) for row in rows]
    for alert in alerts:
        logger.log(
            LEVELS.get(alert["severity"], logging.INFO),
            "data quality %s %s %s: %s",
            alert["date"],
            alert["check"],
            alert["severity"],
            alert["detail"],
        )
    url = str(settings.ALERT_WEBHOOK_URL)
    if not alerts or not url:
        return
    body = {"text": _headline(alerts), "alerts": alerts}
    try:
        response = requests.post(url, json=body, timeout=float(settings.ALERT_TIMEOUT))
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("data quality alert webhook failed: %s", type(exc).__name__)


def _headline(alerts: Sequence[dict[str, Any]]) -> str:
    """The count, then each check and severity, worst first; channels showing only text need it.

    The detail stays in the alerts: it can be long and can hold source URLs.
    """
    kinds = Counter((alert["check"], alert["severity"]) for alert in alerts)
    worst_first = sorted(kinds, key=lambda kind: -LEVELS.get(kind[1], logging.INFO))
    named = [
        f"{check} {severity}"
        + (f" ×{kinds[check, severity]}" if kinds[check, severity] > 1 else "")
        for check, severity in worst_first[:HEADLINE_KINDS]
    ]
    if len(worst_first) > HEADLINE_KINDS:
        named.append(f"+{len(worst_first) - HEADLINE_KINDS} more")
    return f"{len(alerts)} data quality alert(s) for {alerts[0]['date']}: {'; '.join(named)}"


def _payload(row: DataQualityLog) -> dict[str, Any]:
    return {
        "date": row.date.isoformat(),
        "check": row.check_name,
        "severity": row.severity,
        "detail": row.detail,
    }
