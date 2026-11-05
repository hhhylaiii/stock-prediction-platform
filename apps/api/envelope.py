"""The success／data／error／meta envelope every API response is wrapped in."""

from typing import Any

from django.core.exceptions import PermissionDenied as DjangoPermissionDenied
from django.http import Http404
from rest_framework import exceptions
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler


def envelope(data: Any, meta: dict[str, Any] | None = None) -> Response:
    return Response({"success": True, "data": data, "error": None, "meta": meta or {}})


def error_body(exc: exceptions.APIException) -> dict[str, Any]:
    """Machine code plus readable message; validation errors also list the failing fields."""
    if isinstance(exc, exceptions.ValidationError):
        fields = exc.detail if isinstance(exc.detail, dict) else {"non_field_errors": exc.detail}
        message = str(exceptions.ValidationError.default_detail)
        return {"code": "invalid", "message": message, "fields": fields}
    code = exc.get_codes()
    return {
        "code": code if isinstance(code, str) else exc.default_code,
        "message": str(exc.detail),
    }


def exception_handler(exc: Exception, context: dict[str, Any]) -> Response | None:
    """DRF's handler with its body swapped for the envelope; unknown errors still raise."""
    if isinstance(exc, Http404):
        exc = exceptions.NotFound(*exc.args)
    elif isinstance(exc, DjangoPermissionDenied):
        exc = exceptions.PermissionDenied(*exc.args)
    response = drf_exception_handler(exc, context)
    if response is None or not isinstance(exc, exceptions.APIException):
        return response
    response.data = {"success": False, "data": None, "error": error_body(exc), "meta": {}}
    return response
