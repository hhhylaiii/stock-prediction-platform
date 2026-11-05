from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_safe


@require_safe
def index(request: HttpRequest) -> HttpResponse:
    """The single page; its three hash-routed views fetch everything from /api."""
    return render(request, "web/index.html")
