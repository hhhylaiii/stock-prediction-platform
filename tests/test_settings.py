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
def test_the_research_schedule_is_off_unless_switched_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_SCHEDULE_ENABLED", raising=False)
    assert _fresh_import("config.settings.dev").RESEARCH_SCHEDULE_ENABLED is False


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("RESEARCH_AT_EVENING", "22:30", "after ETL_AT_CHECK_QUALITY"),
        ("RESEARCH_AT_MORNING", "05:00", "after ETL_AT_US_CLOSE"),
        ("RESEARCH_RETRAIN_BEFORE", "06:00", "after RESEARCH_AT_MORNING"),
    ],
)
def test_the_research_slots_must_follow_the_data_they_wait_for(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ImproperlyConfigured, match=message):
        _fresh_import("config.settings.dev")


@pytest.mark.usefixtures("isolated_env")
def test_the_monthly_retrain_starts_before_nine_with_half_the_memory_free() -> None:
    module = _fresh_import("config.settings.dev")
    assert module.RESEARCH_RETRAIN_BEFORE == "09:00"
    assert module.RESEARCH_RETRAIN_MIN_FREE == 50
    assert (module.RESEARCH_RETRAIN_COUNTDOWN, module.RESEARCH_RETRAIN_RETRIES) == (1800, 6)


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
def test_filings_must_stop_counting_before_the_evening_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FILING_VISIBLE_BEFORE", "22:50")
    monkeypatch.delenv("ETL_AT_CHECK_QUALITY", raising=False)
    with pytest.raises(ImproperlyConfigured, match="FILING_VISIBLE_BEFORE"):
        _fresh_import("config.settings.dev")


@pytest.mark.usefixtures("isolated_env")
def test_the_day_the_exchanges_lead_dividends_must_be_a_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DIVIDEND_EXCHANGE_PRIMARY_FROM", raising=False)
    assert _fresh_import("config.settings.dev").DIVIDEND_EXCHANGE_PRIMARY_FROM == "2026-10-31"
    monkeypatch.setenv("DIVIDEND_EXCHANGE_PRIMARY_FROM", "2026/10/31")
    with pytest.raises(ImproperlyConfigured, match="DIVIDEND_EXCHANGE_PRIMARY_FROM"):
        _fresh_import("config.settings.dev")


@pytest.mark.usefixtures("isolated_env")
def test_the_statement_scan_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MOPS_BASE_URL", "STATEMENT_SCAN_DAYS_AFTER", "ETL_AT_STATEMENTS"):
        monkeypatch.delenv(name, raising=False)
    module = _fresh_import("config.settings.dev")
    assert module.MOPS_BASE_URL == "https://mopsov.twse.com.tw"
    assert module.STATEMENT_SCAN_DAYS_AFTER == 45
    assert module.ETL_AT["STATEMENTS"] == "19:00"


@pytest.mark.usefixtures("isolated_env")
def test_the_revenue_scan_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("REVENUE_SCAN_DAYS_AFTER", "ETL_AT_REVENUE", "FILING_VISIBLE_BEFORE"):
        monkeypatch.delenv(name, raising=False)
    module = _fresh_import("config.settings.dev")
    assert module.REVENUE_SCAN_DAYS_AFTER == 20
    assert (module.ETL_AT["REVENUE"], module.ETL_AT["REVENUE_LATE"]) == ("19:00", "22:40")


@pytest.mark.usefixtures("isolated_env")
def test_doc_twse_is_asked_six_seconds_apart_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOPS_MIN_INTERVAL", raising=False)
    module = _fresh_import("config.settings.dev")
    assert module.ETL_RATE_LIMITS["mops"]["min_interval"] == 6.0


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


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("value", [None, ""])
def test_feature_parquet_defaults_under_artifacts(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("FEATURE_PARQUET_DIR", raising=False)
    else:
        monkeypatch.setenv("FEATURE_PARQUET_DIR", value)
    module = _fresh_import("config.settings.base")
    assert module.FEATURE_PARQUET_DIR == module.BASE_DIR / "artifacts" / "features"


@pytest.mark.usefixtures("isolated_env")
def test_feature_parquet_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "features"
    monkeypatch.setenv("FEATURE_PARQUET_DIR", str(target))
    module = _fresh_import("config.settings.base")
    assert target == module.FEATURE_PARQUET_DIR


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("value", [None, ""])
def test_model_dir_defaults_under_artifacts(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("MODEL_DIR", raising=False)
    else:
        monkeypatch.setenv("MODEL_DIR", value)
    module = _fresh_import("config.settings.base")
    assert module.MODEL_DIR == module.BASE_DIR / "artifacts" / "models"


@pytest.mark.usefixtures("isolated_env")
def test_model_dir_follows_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "models"
    monkeypatch.setenv("MODEL_DIR", str(target))
    module = _fresh_import("config.settings.base")
    assert target == module.MODEL_DIR
