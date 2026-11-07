"""The Celery application: ``celery -A evilflowers_text.worker worker``.

Task names and queues are a contract with their callers and must not change
(prefix ``evilflowers_text_worker.``):

================  ===============  ==================================================
Task              Queue            Sent by
================  ===============  ==================================================
process_pdf       ``TEXT_QUEUE``   catalog, on upload (``args=[source, entry_id]``)
delete_document   ``TEXT_QUEUE``   (no caller today)
stats             ``TEXT_QUEUE``   (no caller today)
tables            ``TEXT_QUEUE``   analyzer (kwargs)
paragraphs        ``TEXT_QUEUE``   analyzer (kwargs)
embed_query       ``EMBED_QUEUE``  catalog reader (``args=[text, model]``, waits)
================  ===============  ==================================================
"""

from celery import Celery

from evilflowers_text.observability.signals import connect
from evilflowers_text.settings import Settings, get_settings

TASK_PREFIX = "evilflowers_text_worker"


class CeleryConfig:
    """Shared with the sibling extraction services (analyzer, image, equation) where it can be."""

    def __init__(self, settings: Settings):
        self.broker_url = settings.redis_url
        # `embed_query` is awaited by the catalog (`result.get`), so results are stored.
        self.result_backend = settings.redis_url
        self.result_expires = settings.result_expires
        self.broker_connection_retry_on_startup = True
        # Extraction jobs acknowledge late; Redis must not redeliver a book still being read.
        self.broker_transport_options = {"visibility_timeout": settings.broker_visibility_timeout}

        self.task_serializer = "json"
        self.result_serializer = "json"
        self.accept_content = ["json"]
        self.timezone = "Europe/Bratislava"
        self.enable_utc = True

        self.task_default_queue = settings.queue
        self.task_routes = {
            f"{TASK_PREFIX}.embed_query": {"queue": settings.embed_queue},
            f"{TASK_PREFIX}.*": {"queue": settings.queue},
        }
        self.task_track_started = True
        self.worker_prefetch_multiplier = 1
        # Monitoring (Flower, `celery events`).
        self.worker_send_task_events = True
        self.task_send_sent_event = True


def create_app(settings: Settings | None = None) -> Celery:
    settings = settings or get_settings()
    app = Celery(
        "evilflowers_text",
        include=[
            "evilflowers_text.tasks.extraction",
            "evilflowers_text.tasks.embedding",
            "evilflowers_text.tasks.indexing",
        ],
    )
    app.config_from_object(CeleryConfig(settings))
    return app


# Traces, metrics and JSON logs; off unless the pod points them at a collector.
connect()
app = create_app()
