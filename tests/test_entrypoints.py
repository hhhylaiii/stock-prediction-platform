from django.conf import settings

from config import celery_app


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
