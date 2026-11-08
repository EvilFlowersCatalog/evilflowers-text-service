"""The Celery application: ``celery -A evilflowers_text.worker worker``.

Task names and queues are a contract with their callers and must not change
(prefix ``evilflowers_text_worker.``):

================  ===============  ==================================================
Task              Queue            Sent by
================  ===============  ==================================================
tables            ``TEXT_QUEUE``   analyzer (kwargs)
paragraphs        ``TEXT_QUEUE``   analyzer (kwargs)
================  ===============  ==================================================

Nobody waits for a task's result (a job reports to the catalog), so there is no
result backend.
"""

from celery import Celery

from evilflowers_text.observability.signals import connect
from evilflowers_text.settings import Settings, get_settings

TASK_PREFIX = "evilflowers_text_worker"


class CeleryConfig:
    """Shared with the sibling extraction services (analyzer, image, equation) where it can be."""

    def __init__(self, settings: Settings):
        self.broker_url = settings.redis_url
        self.broker_connection_retry_on_startup = True
        # Extraction jobs acknowledge late; Redis must not redeliver a book still being read.
        self.broker_transport_options = {"visibility_timeout": settings.broker_visibility_timeout}

        self.task_serializer = "json"
        self.result_serializer = "json"
        self.accept_content = ["json"]
        self.timezone = "Europe/Bratislava"
        self.enable_utc = True

        self.task_default_queue = settings.queue
        self.task_routes = {f"{TASK_PREFIX}.*": {"queue": settings.queue}}
        self.worker_prefetch_multiplier = 1
        # Monitoring (Flower, `celery events`).
        self.worker_send_task_events = True
        self.task_send_sent_event = True


def create_app(settings: Settings | None = None) -> Celery:
    settings = settings or get_settings()
    app = Celery("evilflowers_text", include=["evilflowers_text.tasks.extraction"])
    app.config_from_object(CeleryConfig(settings))
    return app


# Traces, metrics and JSON logs; off unless the pod points them at a collector.
connect()
app = create_app()
