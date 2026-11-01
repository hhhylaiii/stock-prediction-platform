"""Settings shared by every environment; values come from environment variables."""

from pathlib import Path

import environ
from celery.schedules import crontab

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
STATIC_ROOT = BASE_DIR / "staticfiles"

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


def _at(name: str, default: str, **fields: str) -> crontab:
    """A crontab in TIME_ZONE for the HH:MM in ETL_AT_<name>, falling back to the default."""
    hour, minute = env.str(f"ETL_AT_{name}", default=default).split(":")
    return crontab(hour=hour, minute=minute, **fields)


CELERY_BEAT_SCHEDULE: dict[str, dict[str, object]] = {
    "etl-prepare-day": {"task": "etl.prepare_day", "schedule": _at("PREPARE_DAY", "15:10")},
    "etl-prices": {
        "task": "etl.run_batch",
        "schedule": _at("PRICES", "15:30"),
        "args": ("prices",),
    },
    "etl-refresh-master": {
        "task": "etl.refresh_master",
        "schedule": _at("REFRESH_MASTER", "16:00"),
    },
    "etl-afternoon": {
        "task": "etl.run_batch",
        "schedule": _at("AFTERNOON", "16:30"),
        "args": ("afternoon",),
    },
    "etl-qfii": {"task": "etl.run_batch", "schedule": _at("QFII", "18:30"), "args": ("qfii",)},
    "etl-night": {"task": "etl.run_batch", "schedule": _at("NIGHT", "21:30"), "args": ("night",)},
    "etl-notices-again": {
        "task": "etl.run_batch",
        "schedule": _at("NOTICES_AGAIN", "22:40"),
        "args": ("notices",),
    },
    "etl-check-quality": {"task": "etl.check_quality", "schedule": _at("CHECK_QUALITY", "22:50")},
    "etl-us-close": {"task": "etl.us_close", "schedule": _at("US_CLOSE", "05:30")},
    "etl-weekly-holding": {
        "task": "etl.weekly_holding",
        "schedule": _at("WEEKLY_HOLDING", "10:00", day_of_week="sat"),
    },
    "etl-business-signal": {
        "task": "etl.business_signal",
        "schedule": _at("BUSINESS_SIGNAL", "10:00", day_of_month="15"),
    },
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

RESEARCH_CACHE_DIR = Path(
    env.str("RESEARCH_CACHE_DIR", default="") or BASE_DIR / "artifacts" / "predictions"
)
FEATURE_PARQUET_DIR = Path(
    env.str("FEATURE_PARQUET_DIR", default="") or BASE_DIR / "artifacts" / "features"
)
MODEL_DIR = Path(env.str("MODEL_DIR", default="") or BASE_DIR / "artifacts" / "models")

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
