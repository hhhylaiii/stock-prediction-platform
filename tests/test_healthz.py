from unittest.mock import patch

import pytest
from django.db.utils import OperationalError
from django.test import Client


@pytest.mark.django_db
def test_healthz_ok_when_database_reachable(client: Client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_healthz_reports_503_when_database_down(client: Client) -> None:
    with patch(
        "config.health.connection.ensure_connection",
        side_effect=OperationalError("down"),
    ):
        response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json() == {"status": "error", "database": "unreachable"}


def test_healthz_rejects_non_get(client: Client) -> None:
    response = client.post("/healthz")
    assert response.status_code == 405
