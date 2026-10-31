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


@pytest.mark.usefixtures("isolated_env")
def test_prod_settings_require_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DJANGO_SECRET_KEY", raising=False)
    monkeypatch.setenv("DJANGO_ALLOWED_HOSTS", "example.test")
    with pytest.raises(ImproperlyConfigured, match="DJANGO_SECRET_KEY"):
        _fresh_import("config.settings.prod")


@pytest.mark.usefixtures("isolated_env")
def test_prod_settings_require_allowed_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DJANGO_SECRET_KEY", "x" * 50)
    monkeypatch.delenv("DJANGO_ALLOWED_HOSTS", raising=False)
    with pytest.raises(ImproperlyConfigured, match="DJANGO_ALLOWED_HOSTS"):
        _fresh_import("config.settings.prod")


@pytest.mark.usefixtures("isolated_env")
def test_prod_settings_load_with_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DJANGO_SECRET_KEY", "x" * 50)
    monkeypatch.setenv("DJANGO_ALLOWED_HOSTS", "example.test")
    module = _fresh_import("config.settings.prod")
    assert module.DEBUG is False
    assert module.ALLOWED_HOSTS == ["example.test"]
    assert module.SECRET_KEY == "x" * 50
