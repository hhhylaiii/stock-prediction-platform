"""HTTP access shared by every source: pacing, error mapping and raw storage (SPEC §8.10)."""

from collections.abc import Mapping
from typing import Any, ClassVar

import requests
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from apps.etl.errors import SchemaChangedError, SourceRequestError
from apps.etl.models import RawRecord
from apps.etl.ratelimit import RateLimiter, RetryableError, ThrottledError
from apps.etl.raw import store_raw


def _reject_constant(token: str) -> Any:
    """NaN and Infinity are not JSON; letting them through would fail later in the raw table."""
    raise SchemaChangedError(f"source sent the non-JSON number {token}")


class SourceClient:
    """One source behind its rate limiter; JSON answers must be objects, files come as bytes."""

    source: ClassVar[str]
    label: ClassVar[str]
    raw_model: ClassVar[type[RawRecord]]
    min_interval: ClassVar[float] = 0.0
    throttle_statuses: ClassVar[frozenset[int]] = frozenset({429})

    def __init__(
        self,
        limiter: RateLimiter,
        base_url: str,
        timeout: float,
        session: requests.Session | None = None,
    ) -> None:
        if limiter.limit.min_interval < self.min_interval:
            raise ImproperlyConfigured(
                f"{self.label} requests must be at least {self.min_interval:g} seconds apart"
            )
        self.limiter = limiter
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()

    def get_json(self, path: str, params: Mapping[str, str]) -> dict[str, Any]:
        return self.limiter.call(lambda: self._send("GET", path, params=dict(params)))

    def post_json(self, path: str, data: Mapping[str, str]) -> dict[str, Any]:
        return self.limiter.call(lambda: self._send("POST", path, data=dict(data)))

    def get_content(self, url: str, params: Mapping[str, str]) -> bytes:
        """The body of a GET to an absolute URL, for sources that answer with files."""
        return self.limiter.call(lambda: self._receive("GET", url, params=dict(params)).content)

    def save_raw(self, dataset: str, params: Mapping[str, Any], payload: Any) -> RawRecord:
        return store_raw(self.raw_model, dataset, params, payload)

    def headers(self) -> dict[str, str]:
        return {"User-Agent": str(settings.ETL_USER_AGENT)}

    def check_payload(self, payload: dict[str, Any]) -> None:
        """Hook for sources that report errors inside a 200 response."""

    def _send(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._receive(method, f"{self.base_url}{path}", **kwargs)
        try:
            payload = response.json(parse_constant=_reject_constant)
        except ValueError:
            raise SchemaChangedError(
                f"{self.label} returned non-JSON: {response.text[:200]!r}"
            ) from None
        if not isinstance(payload, dict):
            raise SchemaChangedError(f"{self.label} returned JSON that is not an object")
        self.check_payload(payload)
        return payload

    def _receive(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        try:
            response = self.session.request(
                method, url, timeout=self.timeout, headers=self.headers(), **kwargs
            )
        except requests.RequestException as exc:
            raise RetryableError(f"{self.label} request failed: {exc}") from exc
        status = response.status_code
        if status in self.throttle_statuses:
            raise ThrottledError(f"{self.label} returned {status}")
        if status >= 500:
            raise RetryableError(f"{self.label} returned {status}")
        if status >= 400:
            raise SourceRequestError(f"{self.label} refused the request with {status}")
        return response
