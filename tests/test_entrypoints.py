import os
import subprocess
import sys
from pathlib import Path

import pytest
from django.conf import settings

from config import celery_app

REPO_ROOT = Path(__file__).resolve().parents[1]
PRINT_SETTINGS = "from django.conf import settings; print(settings.SETTINGS_MODULE)"


def test_wsgi_application_is_callable() -> None:
    from config.wsgi import application

    assert callable(application)


def test_asgi_application_is_callable() -> None:
    from config.asgi import application

    assert callable(application)


def test_celery_reads_django_settings() -> None:
    assert str(celery_app.conf.timezone) == "Asia/Taipei"
    assert celery_app.conf.broker_url == settings.CELERY_BROKER_URL
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.worker_prefetch_multiplier == 1


@pytest.mark.parametrize(
    "command",
    [
        ["manage.py", "shell", "-c", PRINT_SETTINGS],
        ["-c", f"import config.wsgi; {PRINT_SETTINGS}"],
        ["-c", f"import config.asgi; {PRINT_SETTINGS}"],
        ["-c", f"import config.celery, django; django.setup(); {PRINT_SETTINGS}"],
    ],
    ids=["manage", "wsgi", "asgi", "celery"],
)
def test_entrypoints_default_to_dev_settings(command: list[str], tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "DJANGO_SETTINGS_MODULE"}
    env["DJANGO_ENV_FILE"] = str(tmp_path / "absent.env")
    result = subprocess.run(
        [sys.executable, *command],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "config.settings.dev"
