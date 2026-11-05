from unittest.mock import ANY

import pytest
from django.core.exceptions import PermissionDenied as DjangoPermissionDenied
from django.http import Http404
from django.test import Client, override_settings
from rest_framework.request import Request
from rest_framework.test import APIRequestFactory

from apps.api.envelope import exception_handler
from apps.api.pagination import EnvelopePagination
from tests.api.conftest import envelope_of

pytestmark = pytest.mark.django_db


def test_unknown_api_path_is_a_not_found_envelope(client: Client) -> None:
    response = client.get("/api/no-such-endpoint")

    assert response.status_code == 404
    assert envelope_of(response) == {
        "success": False,
        "data": None,
        "error": {"code": "not_found", "message": ANY},
        "meta": {},
    }


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete", "options"])
@pytest.mark.parametrize("path", ["/api/etl/runs", "/api/stocks/2330", "/api/data-quality"])
def test_only_reads_are_allowed(client: Client, method: str, path: str) -> None:
    response = getattr(client, method)(path)

    assert response.status_code == 405
    assert response["Allow"] == "GET, HEAD"
    body = envelope_of(response)
    assert body["success"] is False
    assert body["error"]["code"] == "method_not_allowed"


@pytest.mark.parametrize("method", ["get", "head", "post", "options"])
def test_unknown_paths_are_not_found_for_every_method(client: Client, method: str) -> None:
    response = getattr(client, method)("/api/no-such-endpoint")

    assert response.status_code == 404


def test_invalid_query_lists_the_failing_fields(client: Client) -> None:
    response = client.get("/api/data-quality", {"days": "0"})

    assert response.status_code == 400
    body = envelope_of(response)
    assert body["data"] is None
    assert body["error"]["code"] == "invalid"
    assert set(body["error"]["fields"]) == {"days"}


def test_browsers_get_json_not_the_browsable_api(client: Client) -> None:
    response = client.get("/api/no-such-endpoint", HTTP_ACCEPT="text/html,*/*;q=0.8")

    assert response["Content-Type"] == "application/json"


@override_settings(API_THROTTLE_RATE="2/min")
def test_throttle_answers_429_with_retry_after(client: Client) -> None:
    statuses = [client.get("/api/no-such-endpoint").status_code for _ in range(3)]

    assert statuses == [404, 404, 429]
    response = client.get("/api/no-such-endpoint")
    assert envelope_of(response)["error"]["code"] == "throttled"
    assert int(response["Retry-After"]) > 0


@override_settings(API_THROTTLE_RATE="1/min")
def test_throttle_counts_each_client_address_alone(client: Client) -> None:
    first = client.get("/api/no-such-endpoint", REMOTE_ADDR="10.0.0.1")
    other = client.get("/api/no-such-endpoint", REMOTE_ADDR="10.0.0.2")
    spoofed = client.get(
        "/api/no-such-endpoint", REMOTE_ADDR="10.0.0.1", HTTP_X_FORWARDED_FOR="10.0.0.3"
    )

    assert [first.status_code, other.status_code, spoofed.status_code] == [404, 404, 429]


def test_django_permission_denied_becomes_a_403_envelope() -> None:
    response = exception_handler(DjangoPermissionDenied("closed"), {})

    assert response is not None
    assert response.status_code == 403
    assert response.data["error"] == {"code": "permission_denied", "message": "closed"}


def test_django_http404_becomes_a_not_found_envelope() -> None:
    response = exception_handler(Http404("gone"), {})

    assert response is not None
    assert response.status_code == 404
    assert response.data["error"] == {"code": "not_found", "message": "gone"}


def test_unexpected_errors_are_left_to_raise() -> None:
    assert exception_handler(ValueError("bug"), {}) is None


def test_pagination_meta_needs_a_page_first() -> None:
    with pytest.raises(RuntimeError, match="paginate_queryset"):
        EnvelopePagination().meta()


def test_paginated_response_is_an_envelope() -> None:
    paginator = EnvelopePagination()
    request = Request(APIRequestFactory().get("/api/x", {"page": "2", "page_size": "2"}))

    page = paginator.paginate_queryset(list(range(5)), request)
    response = paginator.get_paginated_response(page)

    assert response.data == {
        "success": True,
        "data": [2, 3],
        "error": None,
        "meta": {"total": 5, "page": 2, "page_size": 2, "pages": 3},
    }
