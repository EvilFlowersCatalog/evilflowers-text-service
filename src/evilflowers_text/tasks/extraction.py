"""IP-018 extraction jobs, dispatched by evilflowers-analyzer-service with keyword arguments.

``**_`` keeps the worker compatible with arguments a newer analyzer adds.

The jobs acknowledge late: a worker that dies mid-book, or is stopped (SIGTERM
requeues the job, see `worker.requeue_on_sigterm`), has the message delivered
again. A copy of a job reads its run first and stops unless the run still waits
for it (`jobs.ExtractionJob.run`, issue #12), so it costs one request, not a
second pass over the book.

Under the prefork pool a job runs in a child process. If the child is killed
(an OOM on a large book), the parent sees `WorkerLostError`, and with the
defaults it would acknowledge the message: the job would never report and its
run would wait forever. So such a job is sent again up to `LOST_RETRIES` times.
After that, `report_lost_job` reports it `failed`, and a book that always kills
the worker cannot loop.
"""

from collections.abc import Callable
from typing import Any

from billiard.exceptions import WorkerLostError
from celery import shared_task, signals

from evilflowers_text.jobs import ExtractionJob, Outcome, ParagraphsJob, TablesJob

#: Times a job whose worker process died is sent again before it is reported failed.
LOST_RETRIES = 2
#: Seconds before such a retry.
LOST_RETRY_DELAY = 60

_LOST = {"retry_on_worker_lost": True, "max_retries": LOST_RETRIES, "default_retry_delay": LOST_RETRY_DELAY}


def _tables(run_id: str, source: str, output_dir: str, regions: list[dict], **_) -> TablesJob:
    return TablesJob(run_id, source, output_dir, regions)


def _paragraphs(run_id: str, source: str, **_) -> ParagraphsJob:
    return ParagraphsJob(run_id, source)


#: Task name → the job its keyword arguments describe.
JOBS: dict[str, Callable[..., ExtractionJob]] = {
    "evilflowers_text_worker.tables": _tables,
    "evilflowers_text_worker.paragraphs": _paragraphs,
}


@shared_task(name="evilflowers_text_worker.tables", acks_late=True, **_LOST)
def tables(run_id: str, source: str, output_dir: str, regions: list[dict], **_) -> Outcome:
    """Table regions found by the analyzer's layout pass: markdown, printed caption, crop and position."""
    return _tables(run_id, source, output_dir, regions).run()


@shared_task(name="evilflowers_text_worker.paragraphs", acks_late=True, **_LOST)
def paragraphs(run_id: str, source: str, **_) -> Outcome:
    """The paragraph index (IP-018 Phase 2b): paragraphs, sentences, embeddings and links."""
    return _paragraphs(run_id, source).run()


def report_lost_job(sender: Any = None, exception: BaseException | None = None, kwargs: dict | None = None, **_):
    """In the parent: the job's worker process died on every try, so report the job failed."""
    factory = JOBS.get(getattr(sender, "name", ""))
    if factory is None or not isinstance(exception, WorkerLostError):
        return
    factory(**(kwargs or {})).report_failure(exception)


signals.task_failure.connect(report_lost_job, weak=False)
