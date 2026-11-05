from django.urls import path, re_path

from apps.api.ops import DataQualityView, EtlRunListView, NotFoundView
from apps.api.predictions import PredictionListView
from apps.api.status import StatusView
from apps.api.stock_features import StockFeaturesView
from apps.api.stock_history import StockHistoryView
from apps.api.stock_predictions import StockPredictionsView
from apps.api.stocks import StockDetailView, StockPricesView

app_name = "api"

STOCK = r"^stocks/(?P<stock_id>[0-9A-Za-z]{1,10})"

urlpatterns = [
    re_path(STOCK + r"$", StockDetailView.as_view(), name="stock-detail"),
    re_path(STOCK + r"/prices$", StockPricesView.as_view(), name="stock-prices"),
    re_path(STOCK + r"/predictions$", StockPredictionsView.as_view(), name="stock-predictions"),
    re_path(STOCK + r"/features$", StockFeaturesView.as_view(), name="stock-features"),
    re_path(STOCK + r"/history$", StockHistoryView.as_view(), name="stock-history"),
    path("status", StatusView.as_view(), name="status"),
    path("predictions", PredictionListView.as_view(), name="predictions"),
    path("etl/runs", EtlRunListView.as_view(), name="etl-runs"),
    path("data-quality", DataQualityView.as_view(), name="data-quality"),
    re_path(r"", NotFoundView.as_view(), name="not-found"),
]
