"""IP-018 extraction jobs, dispatched by evilflowers-analyzer-service with keyword arguments.

Each task body is the kit's ``runtime.run_job`` (issue #16): it validates the keyword arguments against the shared
contract (``RegionsJob`` / ``ParagraphsJob``; unknown ones are ignored, so a newer analyzer can add some), counts the
delivery and runs the job. An invalid message is reported failed and never retried.

``JOB_TASK_OPTIONS``: the job is acknowledged after it ends, and requeued if its worker process dies (an OOM); the
delivery cap (``MAX_DELIVERIES``) then reports a job that kills its worker every time as failed instead of looping.
On SIGTERM the kit's ``shutdown`` gives the job ``WORKER_SOFT_SHUTDOWN_TIMEOUT`` to finish, then requeues it (#17).
"""

from typing import Any

from celery import Task, shared_task

# The app's module, so `create_app` has run in this process: a pool child started by *spawn* (not fork) imports only
# the tasks module, and `runtime()` would find nothing set up there.
from evilflowers_text import worker as _worker  # noqa: F401  # isort: skip
from evilflowers_text.jobs import ParagraphsJob, TablesJob
from evilflowers_worker import contract
from evilflowers_worker.runtime import JOB_TASK_OPTIONS, run_job


@shared_task(name="evilflowers_text_worker.tables", bind=True, **JOB_TASK_OPTIONS)
def tables(self: Task, **kwargs: Any) -> str:
    """Table regions found by the analyzer's layout pass: markdown, printed caption, crop and position."""
    return run_job(self, TablesJob, contract.RegionsJob, kwargs)


@shared_task(name="evilflowers_text_worker.paragraphs", bind=True, **JOB_TASK_OPTIONS)
def paragraphs(self: Task, **kwargs: Any) -> str:
    """The paragraph index (IP-018 Phase 2b): paragraphs, sentences, embeddings and links."""
    return run_job(self, ParagraphsJob, contract.ParagraphsJob, kwargs)
