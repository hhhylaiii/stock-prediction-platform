"""Alerts for data-quality warnings and errors: always logged, posted when a webhook is set."""

import logging
from collections.abc import Sequence
from typing import Any

import requests
from django.conf import settings

from apps.etl.models import DataQualityLog
from apps.etl.quality import alerting

logger = logging.getLogger(__name__)


def dispatch(results: Sequence[DataQualityLog]) -> None:
    """Log each warning or error; a failing webhook is logged, never raised."""
    alerts = [_payload(row) for row in alerting(results)]
    for alert in alerts:
        level = (
            logging.ERROR if alert["severity"] == DataQualityLog.Severity.ERROR else logging.WARNING
        )
        logger.log(
            level,
            "data quality %s %s %s: %s",
            alert["date"],
            alert["check"],
            alert["severity"],
            alert["detail"],
        )
    url = str(settings.ALERT_WEBHOOK_URL)
    if not alerts or not url:
        return
    body = {
        "text": f"{len(alerts)} data quality alert(s) for {alerts[0]['date']}",
        "alerts": alerts,
    }
    try:
        response = requests.post(url, json=body, timeout=float(settings.ALERT_TIMEOUT))
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("data quality alert webhook failed: %s", type(exc).__name__)


def _payload(row: DataQualityLog) -> dict[str, Any]:
    return {
        "date": row.date.isoformat(),
        "check": row.check_name,
        "severity": row.severity,
        "detail": row.detail,
    }
