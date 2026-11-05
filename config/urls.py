from django.contrib import admin
from django.urls import include, path

from config.health import healthz

urlpatterns = [
    path("healthz", healthz, name="healthz"),
    path("admin/", admin.site.urls),
    path("api/", include("apps.api.urls")),
]
