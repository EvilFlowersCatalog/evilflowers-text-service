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
``evilflowers_text_worker``). Without ``--queues`` on the command line, that is.

Nobody waits for a task's result (a job reports to the catalog), so there is no
result backend.
"""

import logging

from celery import Celery, signals
from celery.apps.worker import install_worker_term_hard_handler
from celery.concurrency.solo import TaskPool as SoloPool
from kombu import Exchange, Queue

from evilflowers_text.observability.signals import connect
from evilflowers_text.settings import Settings, get_settings

TASK_PREFIX = "evilflowers_text_worker"

logger = logging.getLogger(__name__)


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

        # Each queue on its own direct exchange of the same name: what Celery declares for a queue it
        # only knows by name, as the analyzer's `apply_async(queue=...)` does. Sharing one exchange
        # would leave the analyzer's messages without a binding.
        self.task_queues = [
            Queue(name, Exchange(name, type="direct"), routing_key=name) for name in settings.consumed_queues
        ]
        self.task_default_queue = settings.consumed_queues[0]
        self.task_routes = {
            f"{TASK_PREFIX}.tables": {"queue": settings.tables_queue},
            f"{TASK_PREFIX}.paragraphs": {"queue": settings.paragraphs_queue},
        }
        self.worker_prefetch_multiplier = 1
        # SIGTERM (a scale-in, a rollout) puts the running job's message back on its queue after
        # this many seconds, instead of after the visibility timeout (12 h). See `requeue_on_sigterm`.
        self.worker_soft_shutdown_timeout = settings.worker_soft_shutdown_timeout
        self.worker_enable_soft_shutdown_on_idle = settings.worker_enable_soft_shutdown_on_idle
        # Monitoring (Flower, `celery events`).
        self.worker_send_task_events = True
        self.task_send_sent_event = True


def requeue_on_sigterm(sender, **_) -> None:
    """Make SIGTERM a cold shutdown with a soft-shutdown grace (issue #13).

    Celery's SIGTERM is a warm shutdown: it waits for the running job, Kubernetes kills the pod
    at its grace period, and the unacknowledged message returns only after the visibility timeout.
    A cold shutdown waits `worker_soft_shutdown_timeout`, then cancels the job and Redis gets the
    message back at once. This is what `REMAP_SIGTERM=SIGQUIT` does, chosen here per pool:

    - The solo pool cannot cancel a task. Its cancel raises inside the job, the job fails, and
      `acks_late` then acknowledges the message, which is lost. So under solo SIGTERM stays warm.
    - Prefork (`--concurrency=1` keeps one job at a time) cancels the child, and the message goes back.
    """
    if isinstance(sender.pool, SoloPool):
        logger.warning("--pool=solo: SIGTERM does not requeue a running job; it returns after the visibility timeout")
        return
    install_worker_term_hard_handler(sender.controller, sig="SIGTERM")


def create_app(settings: Settings | None = None) -> Celery:
    settings = settings or get_settings()
    app = Celery("evilflowers_text", include=["evilflowers_text.tasks.extraction"])
    app.config_from_object(CeleryConfig(settings))
    return app


# Traces, metrics and JSON logs; off unless the pod points them at a collector.
connect()
signals.worker_ready.connect(requeue_on_sigterm, weak=False)
app = create_app()
