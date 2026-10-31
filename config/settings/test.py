"""Settings for the pytest run; the database comes from DATABASE_URL like other envs."""

from .base import *  # noqa: F403

SECRET_KEY = "test-only-insecure-key"
ALLOWED_HOSTS = ["testserver", "localhost"]
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True
