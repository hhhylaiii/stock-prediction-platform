from django.urls import path, re_path

from apps.api.ops import DataQualityView, EtlRunListView, NotFoundView
from apps.api.stocks import StockDetailView, StockPricesView

app_name = "api"

STOCK = r"^stocks/(?P<stock_id>[0-9A-Za-z]{1,10})"

urlpatterns = [
    re_path(STOCK + r"$", StockDetailView.as_view(), name="stock-detail"),
    re_path(STOCK + r"/prices$", StockPricesView.as_view(), name="stock-prices"),
    path("etl/runs", EtlRunListView.as_view(), name="etl-runs"),
    path("data-quality", DataQualityView.as_view(), name="data-quality"),
    re_path(r"", NotFoundView.as_view(), name="not-found"),
]
