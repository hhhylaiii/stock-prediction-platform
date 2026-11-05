from typing import TYPE_CHECKING

from django.conf import settings
from rest_framework.request import Request
from rest_framework.throttling import SimpleRateThrottle

if TYPE_CHECKING:
    from rest_framework.views import APIView


class ApiThrottle(SimpleRateThrottle):
    """One shared budget per client address at settings.API_THROTTLE_RATE, read per request."""

    scope = "api"

    def get_rate(self) -> str:
        return str(settings.API_THROTTLE_RATE)

    def get_cache_key(self, request: Request, view: "APIView") -> str:
        return self.cache_format % {"scope": self.scope, "ident": self.get_ident(request)}
