from typing import Any

from django.conf import settings
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

from apps.api.envelope import envelope


class EnvelopePagination(PageNumberPagination):
    """Page-number pages of API_PAGE_SIZE; ?page_size= may shrink or grow up to the cap."""

    page_size_query_param = "page_size"

    def __init__(self) -> None:
        self.page_size = settings.API_PAGE_SIZE
        self.max_page_size = settings.API_MAX_PAGE_SIZE

    def meta(self) -> dict[str, int]:
        page = getattr(self, "page", None)
        if page is None:
            raise RuntimeError("paginate_queryset must run before meta")
        paginator = page.paginator
        return {
            "total": paginator.count,
            "page": page.number,
            "page_size": paginator.per_page,
            "pages": paginator.num_pages,
        }

    def get_paginated_response(self, data: Any) -> Response:
        return envelope(data, self.meta())
