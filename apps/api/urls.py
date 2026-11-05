from django.urls import path, re_path

from apps.api.ops import DataQualityView, EtlRunListView, NotFoundView

app_name = "api"

urlpatterns = [
    path("etl/runs", EtlRunListView.as_view(), name="etl-runs"),
    path("data-quality", DataQualityView.as_view(), name="data-quality"),
    re_path(r"", NotFoundView.as_view(), name="not-found"),
]
