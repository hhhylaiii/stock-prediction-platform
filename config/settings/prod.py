"""Settings for the always-on Compose deployment; secrets must come from the environment."""

from .base import *  # noqa: F403
from .base import env

DEBUG = False
SECRET_KEY = env.str("DJANGO_SECRET_KEY")
ALLOWED_HOSTS = env.list("DJANGO_ALLOWED_HOSTS")
SESSION_COOKIE_SECURE = env.bool("DJANGO_SECURE_COOKIES", default=True)
CSRF_COOKIE_SECURE = env.bool("DJANGO_SECURE_COOKIES", default=True)
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
