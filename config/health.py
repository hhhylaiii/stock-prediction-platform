from django.db import connection
from django.db.utils import DatabaseError
from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET


@require_GET
def healthz(request: HttpRequest) -> JsonResponse:
    """Liveness probe for Compose: 200 when the database answers, 503 otherwise."""
    try:
        connection.ensure_connection()
    except DatabaseError:
        return JsonResponse({"status": "error", "database": "unreachable"}, status=503)
    return JsonResponse({"status": "ok", "database": "ok"})
