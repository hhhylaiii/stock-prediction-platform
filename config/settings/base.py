"""Settings shared by every environment; values come from environment variables."""

from datetime import time
from pathlib import Path

import environ
from celery.schedules import crontab
from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent.parent

env = environ.Env()
environ.Env.read_env(env.str("DJANGO_ENV_FILE", default=str(BASE_DIR / ".env")))

SECRET_KEY = env.str("DJANGO_SECRET_KEY", default="")
DEBUG = False
ALLOWED_HOSTS: list[str] = env.list("DJANGO_ALLOWED_HOSTS", default=[])

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "apps.market",
    "apps.etl",
    "apps.research",
    "apps.api",
    "apps.web",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": env.db_url(
        "DATABASE_URL",
        default="postgres://twp:twp@localhost:5432/twp",
    ),
}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "zh-hant"
TIME_ZONE = "Asia/Taipei"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT: Path | None = BASE_DIR / "staticfiles"
WHITENOISE_USE_FINDERS = True

CELERY_BROKER_URL = env.str("CELERY_BROKER_URL", default="redis://localhost:6379/0")
CELERY_RESULT_BACKEND = env.str("CELERY_RESULT_BACKEND", default="")
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_WORKER_CONCURRENCY = env.int("CELERY_WORKER_CONCURRENCY", default=1)
CELERY_BROKER_TRANSPORT_OPTIONS = {
    "visibility_timeout": env.int("CELERY_VISIBILITY_TIMEOUT", default=43200)
}
ETL_NOT_READY_COUNTDOWN = env.int("ETL_NOT_READY_COUNTDOWN", default=1800)
ETL_NOT_READY_RETRIES = env.int("ETL_NOT_READY_RETRIES", default=6)
ETL_CATCHUP_LOOKBACK_DAYS = env.int("ETL_CATCHUP_LOOKBACK_DAYS", default=14)
FILING_VISIBLE_BEFORE = env.str("FILING_VISIBLE_BEFORE", default="22:40")


ETL_AT = {
    name: env.str(f"ETL_AT_{name}", default=default)
    for name, default in (
        ("PREPARE_DAY", "15:10"),
        ("PRICES", "15:30"),
        ("REFRESH_MASTER", "16:00"),
        ("AFTERNOON", "16:30"),
        ("QFII", "18:30"),
        ("NIGHT", "21:30"),
        ("NOTICES_AGAIN", "22:40"),
        ("STATEMENTS", "19:00"),
        ("REVENUE", "19:00"),
        ("CHECK_QUALITY", "22:50"),
        ("US_CLOSE", "05:30"),
        ("WEEKLY_HOLDING", "10:00"),
        ("BUSINESS_SIGNAL", "10:00"),
    )
} | {"STATEMENTS_LATE": FILING_VISIBLE_BEFORE, "REVENUE_LATE": FILING_VISIBLE_BEFORE}
if time.fromisoformat(FILING_VISIBLE_BEFORE) >= time.fromisoformat(ETL_AT["CHECK_QUALITY"]):
    raise ImproperlyConfigured("FILING_VISIBLE_BEFORE must come before ETL_AT_CHECK_QUALITY")


RESEARCH_SCHEDULE_ENABLED = env.bool("RESEARCH_SCHEDULE_ENABLED", default=False)
RESEARCH_AT = {
    name: env.str(f"RESEARCH_AT_{name}", default=default)
    for name, default in (("EVENING", "23:00"), ("MORNING", "06:00"))
}
RESEARCH_US_CLOSE_COUNTDOWN = env.int("RESEARCH_US_CLOSE_COUNTDOWN", default=300)
RESEARCH_US_CLOSE_RETRIES = env.int("RESEARCH_US_CLOSE_RETRIES", default=6)
RESEARCH_LOCK_COUNTDOWN = env.int("RESEARCH_LOCK_COUNTDOWN", default=600)
RESEARCH_LOCK_RETRIES = env.int("RESEARCH_LOCK_RETRIES", default=6)
RESEARCH_RETRAIN_BEFORE = env.str("RESEARCH_RETRAIN_BEFORE", default="09:00")
RESEARCH_RETRAIN_MIN_FREE = env.int("RESEARCH_RETRAIN_MIN_FREE", default=50)
RESEARCH_RETRAIN_COUNTDOWN = env.int("RESEARCH_RETRAIN_COUNTDOWN", default=1800)
RESEARCH_RETRAIN_RETRIES = env.int("RESEARCH_RETRAIN_RETRIES", default=6)
if time.fromisoformat(RESEARCH_AT["EVENING"]) <= time.fromisoformat(ETL_AT["CHECK_QUALITY"]):
    raise ImproperlyConfigured("RESEARCH_AT_EVENING must come after ETL_AT_CHECK_QUALITY")
if time.fromisoformat(RESEARCH_AT["MORNING"]) <= time.fromisoformat(ETL_AT["US_CLOSE"]):
    raise ImproperlyConfigured("RESEARCH_AT_MORNING must come after ETL_AT_US_CLOSE")
if time.fromisoformat(RESEARCH_RETRAIN_BEFORE) <= time.fromisoformat(RESEARCH_AT["MORNING"]):
    raise ImproperlyConfigured("RESEARCH_RETRAIN_BEFORE must come after RESEARCH_AT_MORNING")


def _crontab(at: str, **fields: str) -> crontab:
    """A crontab in TIME_ZONE at an HH:MM."""
    hour, minute = at.split(":")
    return crontab(hour=hour, minute=minute, **fields)


def _at(name: str, **fields: str) -> crontab:
    """A crontab in TIME_ZONE at the HH:MM of ETL_AT[name]."""
    return _crontab(ETL_AT[name], **fields)


CELERY_BEAT_SCHEDULE: dict[str, dict[str, object]] = {
    "etl-prepare-day": {"task": "etl.prepare_day", "schedule": _at("PREPARE_DAY")},
    "etl-prices": {
        "task": "etl.run_batch",
        "schedule": _at("PRICES"),
        "args": ("prices",),
    },
    "etl-refresh-master": {
        "task": "etl.refresh_master",
        "schedule": _at("REFRESH_MASTER"),
    },
    "etl-afternoon": {
        "task": "etl.run_batch",
        "schedule": _at("AFTERNOON"),
        "args": ("afternoon",),
    },
    "etl-qfii": {"task": "etl.run_batch", "schedule": _at("QFII"), "args": ("qfii",)},
    "etl-night": {"task": "etl.run_batch", "schedule": _at("NIGHT"), "args": ("night",)},
    "etl-notices-again": {
        "task": "etl.run_batch",
        "schedule": _at("NOTICES_AGAIN"),
        "args": ("notices",),
    },
    "etl-statements": {
        "task": "etl.statement_scan",
        "schedule": _at("STATEMENTS"),
        "args": ("STATEMENTS",),
    },
    "etl-statements-late": {
        "task": "etl.statement_scan",
        "schedule": _at("STATEMENTS_LATE"),
        "args": ("STATEMENTS_LATE",),
    },
    "etl-revenue": {
        "task": "etl.revenue_scan",
        "schedule": _at("REVENUE"),
        "args": ("REVENUE",),
    },
    "etl-revenue-late": {
        "task": "etl.revenue_scan",
        "schedule": _at("REVENUE_LATE"),
        "args": ("REVENUE_LATE",),
    },
    "etl-check-quality": {"task": "etl.check_quality", "schedule": _at("CHECK_QUALITY")},
    "etl-us-close": {"task": "etl.us_close", "schedule": _at("US_CLOSE")},
    "etl-weekly-holding": {
        "task": "etl.weekly_holding",
        "schedule": _at("WEEKLY_HOLDING", day_of_week="sat"),
    },
    "etl-business-signal": {
        "task": "etl.business_signal",
        "schedule": _at("BUSINESS_SIGNAL", day_of_month="15"),
    },
    "research-evening": {"task": "research.evening", "schedule": _crontab(RESEARCH_AT["EVENING"])},
    "research-morning": {"task": "research.morning", "schedule": _crontab(RESEARCH_AT["MORNING"])},
}

TWSE_BASE_URL = env.str("TWSE_BASE_URL", default="https://www.twse.com.tw")
TWSE_TIMEOUT = env.float("TWSE_TIMEOUT", default=30.0)
ETL_USER_AGENT = env.str(
    "ETL_USER_AGENT", default="stock-prediction-platform/0.1 (personal research)"
)
TPEX_BASE_URL = env.str("TPEX_BASE_URL", default="https://www.tpex.org.tw")
TPEX_TIMEOUT = env.float("TPEX_TIMEOUT", default=30.0)
FINMIND_BASE_URL = env.str("FINMIND_BASE_URL", default="https://api.finmindtrade.com")
FINMIND_TIMEOUT = env.float("FINMIND_TIMEOUT", default=60.0)
FINMIND_API_TOKEN = env.str("FINMIND_API_TOKEN", default="")
TDCC_BASE_URL = env.str("TDCC_BASE_URL", default="https://opendata.tdcc.com.tw")
TDCC_TIMEOUT = env.float("TDCC_TIMEOUT", default=60.0)
YAHOO_BASE_URL = env.str("YAHOO_BASE_URL", default="https://query1.finance.yahoo.com")
YAHOO_TIMEOUT = env.float("YAHOO_TIMEOUT", default=30.0)
DATAGOV_BASE_URL = env.str("DATAGOV_BASE_URL", default="https://data.gov.tw")
DATAGOV_TIMEOUT = env.float("DATAGOV_TIMEOUT", default=60.0)
MOPS_DOC_BASE_URL = env.str("MOPS_DOC_BASE_URL", default="https://doc.twse.com.tw")
MOPS_BASE_URL = env.str("MOPS_BASE_URL", default="https://mopsov.twse.com.tw")
MOPS_TIMEOUT = env.float("MOPS_TIMEOUT", default=60.0)
STATEMENT_SCAN_DAYS_AFTER = env.int("STATEMENT_SCAN_DAYS_AFTER", default=45)
REVENUE_SCAN_DAYS_AFTER = env.int("REVENUE_SCAN_DAYS_AFTER", default=20)
ETL_RATE_LIMITS: dict[str, dict[str, float]] = {
    "twse": {
        "min_interval": env.float("TWSE_MIN_INTERVAL", default=3.5),
        "max_retries": env.int("TWSE_MAX_RETRIES", default=4),
        "backoff_base": env.float("TWSE_BACKOFF_BASE", default=5.0),
        "backoff_max": env.float("TWSE_BACKOFF_MAX", default=120.0),
        "cooldown": env.float("TWSE_COOLDOWN", default=900.0),
    },
    "tpex": {
        "min_interval": env.float("TPEX_MIN_INTERVAL", default=3.5),
        "max_retries": env.int("TPEX_MAX_RETRIES", default=4),
        "backoff_base": env.float("TPEX_BACKOFF_BASE", default=5.0),
        "backoff_max": env.float("TPEX_BACKOFF_MAX", default=120.0),
        "cooldown": env.float("TPEX_COOLDOWN", default=900.0),
    },
    "finmind": {
        "min_interval": float(
            env.str("FINMIND_MIN_INTERVAL", default="") or (6.5 if FINMIND_API_TOKEN else 13.0)
        ),
        "max_retries": env.int("FINMIND_MAX_RETRIES", default=4),
        "backoff_base": env.float("FINMIND_BACKOFF_BASE", default=10.0),
        "backoff_max": env.float("FINMIND_BACKOFF_MAX", default=300.0),
        "cooldown": env.float("FINMIND_COOLDOWN", default=3700.0),
    },
    "tdcc": {
        "min_interval": env.float("TDCC_MIN_INTERVAL", default=3.0),
        "max_retries": env.int("TDCC_MAX_RETRIES", default=4),
        "backoff_base": env.float("TDCC_BACKOFF_BASE", default=5.0),
        "backoff_max": env.float("TDCC_BACKOFF_MAX", default=120.0),
        "cooldown": env.float("TDCC_COOLDOWN", default=900.0),
    },
    "yahoo": {
        "min_interval": env.float("YAHOO_MIN_INTERVAL", default=2.0),
        "max_retries": env.int("YAHOO_MAX_RETRIES", default=4),
        "backoff_base": env.float("YAHOO_BACKOFF_BASE", default=5.0),
        "backoff_max": env.float("YAHOO_BACKOFF_MAX", default=120.0),
        "cooldown": env.float("YAHOO_COOLDOWN", default=900.0),
    },
    "mops": {
        "min_interval": env.float("MOPS_MIN_INTERVAL", default=6.0),
        "max_retries": env.int("MOPS_MAX_RETRIES", default=4),
        "backoff_base": env.float("MOPS_BACKOFF_BASE", default=5.0),
        "backoff_max": env.float("MOPS_BACKOFF_MAX", default=120.0),
        "cooldown": env.float("MOPS_COOLDOWN", default=900.0),
    },
    "datagov": {
        "min_interval": env.float("DATAGOV_MIN_INTERVAL", default=2.0),
        "max_retries": env.int("DATAGOV_MAX_RETRIES", default=4),
        "backoff_base": env.float("DATAGOV_BACKOFF_BASE", default=5.0),
        "backoff_max": env.float("DATAGOV_BACKOFF_MAX", default=120.0),
        "cooldown": env.float("DATAGOV_COOLDOWN", default=900.0),
    },
}
ETL_MAX_CONSECUTIVE_REJECTIONS = env.int("ETL_MAX_CONSECUTIVE_REJECTIONS", default=20)
ALERT_WEBHOOK_URL = env.str("ALERT_WEBHOOK_URL", default="")
ALERT_TIMEOUT = env.float("ALERT_TIMEOUT", default=10.0)

API_PAGE_SIZE = env.int("API_PAGE_SIZE", default=20)
API_MAX_PAGE_SIZE = env.int("API_MAX_PAGE_SIZE", default=100)
API_THROTTLE_RATE = env.str("API_THROTTLE_RATE", default="300/min")
API_FULL_MONTH_DAYS = env.int("API_FULL_MONTH_DAYS", default=10)
API_CORRECTIONS_CACHE_SECONDS = env.int("API_CORRECTIONS_CACHE_SECONDS", default=60)
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "DEFAULT_THROTTLE_CLASSES": ["apps.api.throttling.ApiThrottle"],
    "EXCEPTION_HANDLER": "apps.api.envelope.exception_handler",
    "NUM_PROXIES": env.int("API_NUM_PROXIES", default=0),
}

RESEARCH_CACHE_DIR = Path(
    env.str("RESEARCH_CACHE_DIR", default="") or BASE_DIR / "artifacts" / "predictions"
)
FEATURE_PARQUET_DIR = Path(
    env.str("FEATURE_PARQUET_DIR", default="") or BASE_DIR / "artifacts" / "features"
)
MODEL_DIR = Path(env.str("MODEL_DIR", default="") or BASE_DIR / "artifacts" / "models")
SHUFFLE_IC_LIMIT = env.float("SHUFFLE_IC_LIMIT", default=0.02)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "plain"},
    },
    "root": {"handlers": ["console"], "level": env.str("LOG_LEVEL", default="INFO")},
}
