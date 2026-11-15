"""The Celery application: ``celery -A evilflowers_text.worker worker --pool=prefork --concurrency=1``.

Task names are a contract with their callers and must not change (prefix
``evilflowers_text_worker.``). Each job has its own queue (issue #13), so a slow
table review never holds up paragraphs and each list can be scaled on its own:

================  ================================================  ==================
Task              Queue                                             Sent by
================  ================================================  ==================
tables            ``TABLES_QUEUE`` (evilflowers.text.tables)          analyzer (kwargs)
paragraphs        ``PARAGRAPHS_QUEUE`` (evilflowers.text.paragraphs)  analyzer (kwargs)
================  ================================================  ==================

The analyzer names the queue when it sends, so these routes only cover callers
that do not. The worker consumes ``TEXT_QUEUE`` (default: both, plus the legacy
``evilflowers_text_worker``), unless ``--queues`` is on the command line.

Everything else (delivery semantics, scale-in, telemetry, JSON logs) is the kit's
``runtime.create_app`` (evilflowers-worker ``docs/architecture.md`` §3.3, §4; issues #16, #17).
"""

from celery import Celery

from evilflowers_text.settings import Settings, get_settings
from evilflowers_worker import runtime

SERVICE = "text"
TASK_PREFIX = "evilflowers_text_worker"


def create_app(settings: Settings | None = None) -> Celery:
    settings = settings or get_settings()
    return runtime.create_app(
        SERVICE,
        settings,
        include=["evilflowers_text.tasks.extraction"],
        queues=settings.consumed_queues,
        routes={
            f"{TASK_PREFIX}.tables": settings.tables_queue,
            f"{TASK_PREFIX}.paragraphs": settings.paragraphs_queue,
        },
    )


app = create_app()
