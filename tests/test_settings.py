import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import connection


def _fresh_import(module_name: str) -> ModuleType:
    """Import a settings module from scratch so env changes take effect."""
    for name in [m for m in sys.modules if m.startswith("config.settings.")]:
        del sys.modules[name]
    return importlib.import_module(module_name)


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Ignore any local .env and restore the original settings modules afterwards."""
    saved = {k: v for k, v in sys.modules.items() if k.startswith("config.settings.")}
    monkeypatch.setenv("DJANGO_ENV_FILE", str(tmp_path / "absent.env"))
    yield
    for name in [m for m in sys.modules if m.startswith("config.settings.")]:
        del sys.modules[name]
    sys.modules.update(saved)


def test_test_settings_are_active() -> None:
    assert settings.SETTINGS_MODULE == "config.settings.test"


def test_timezone_is_taipei_and_aware() -> None:
    assert settings.TIME_ZONE == "Asia/Taipei"
    assert settings.USE_TZ is True


def test_all_project_apps_installed() -> None:
    for app in ("apps.market", "apps.etl", "apps.research", "apps.api", "apps.web"):
        assert app in settings.INSTALLED_APPS


@pytest.mark.django_db
def test_database_is_reachable_postgres() -> None:
    connection.ensure_connection()
    assert connection.vendor == "postgresql"
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        assert cursor.fetchone() == (1,)


@pytest.mark.usefixtures("isolated_env")
def test_dev_settings_enable_debug() -> None:
    module = _fresh_import("config.settings.dev")
    assert module.DEBUG is True
    assert module.TIME_ZONE == "Asia/Taipei"


PROD_ENV = {
    "DJANGO_SECRET_KEY": "x" * 50,
    "DJANGO_ALLOWED_HOSTS": "example.test",
    "DATABASE_URL": "postgres://u:p@dbhost:5433/prod",
}


def _set_prod_env(monkeypatch: pytest.MonkeyPatch, **overrides: str | None) -> None:
    for key, value in {**PROD_ENV, **overrides}.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("missing", ["DJANGO_SECRET_KEY", "DJANGO_ALLOWED_HOSTS", "DATABASE_URL"])
def test_prod_settings_require_each_variable(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    _set_prod_env(monkeypatch, **{missing: None})
    with pytest.raises(ImproperlyConfigured, match=missing):
        _fresh_import("config.settings.prod")


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("blank", ["", "   "])
def test_prod_settings_reject_blank_secret_key(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    _set_prod_env(monkeypatch, DJANGO_SECRET_KEY=blank)
    with pytest.raises(ImproperlyConfigured, match="must not be empty"):
        _fresh_import("config.settings.prod")


@pytest.mark.usefixtures("isolated_env")
def test_prod_settings_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_prod_env(monkeypatch)
    module = _fresh_import("config.settings.prod")
    assert module.DEBUG is False
    assert module.ALLOWED_HOSTS == ["example.test"]
    assert module.SECRET_KEY == "x" * 50
    assert module.DATABASES["default"]["HOST"] == "dbhost"
    assert module.DATABASES["default"]["NAME"] == "prod"


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize(
    ("token", "configured", "interval"),
    [("abc", "", 6.5), ("", "", 13.0), ("abc", "10", 10.0)],
)
def test_finmind_interval_follows_the_quota_of_the_token(
    monkeypatch: pytest.MonkeyPatch, token: str, configured: str, interval: float
) -> None:
    monkeypatch.setenv("FINMIND_API_TOKEN", token)
    monkeypatch.setenv("FINMIND_MIN_INTERVAL", configured)
    module = _fresh_import("config.settings.dev")
    assert module.ETL_RATE_LIMITS["finmind"]["min_interval"] == interval


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("value", [None, ""])
def test_research_cache_defaults_under_artifacts(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("RESEARCH_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("RESEARCH_CACHE_DIR", value)
    module = _fresh_import("config.settings.base")
    assert module.RESEARCH_CACHE_DIR == module.BASE_DIR / "artifacts" / "predictions"


@pytest.mark.usefixtures("isolated_env")
def test_research_cache_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("RESEARCH_CACHE_DIR", str(cache))
    module = _fresh_import("config.settings.base")
    assert cache == module.RESEARCH_CACHE_DIR
