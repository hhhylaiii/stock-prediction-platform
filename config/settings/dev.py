"""Local development settings."""

from .base import *  # noqa: F403
from .base import env

DEBUG = True
SECRET_KEY = env.str("DJANGO_SECRET_KEY", default="dev-only-insecure-key")
ALLOWED_HOSTS = env.list("DJANGO_ALLOWED_HOSTS", default=["localhost", "127.0.0.1"])
