"""Settings for the pytest run; the database comes from DATABASE_URL like other envs."""

from .base import *  # noqa: F403

SECRET_KEY = "test-only-insecure-key"
ALLOWED_HOSTS = ["testserver", "localhost"]
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True
FINMIND_API_TOKEN = ""
ALERT_WEBHOOK_URL = ""
ETL_RATE_LIMITS = {
    **ETL_RATE_LIMITS,  # noqa: F405
    "finmind": {**ETL_RATE_LIMITS["finmind"], "min_interval": 13.0},  # noqa: F405
}
