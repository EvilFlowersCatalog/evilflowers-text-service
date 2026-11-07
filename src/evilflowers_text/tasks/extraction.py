"""IP-018 extraction jobs, dispatched by evilflowers-analyzer-service with keyword arguments.

``**_`` keeps the worker compatible with arguments a newer analyzer adds.

The jobs acknowledge late: a worker that dies mid-book has the message
redelivered, and the catalog answers 409 (`RunNotRunning`) to a run that
already finished, so a second delivery is harmless.
"""

from celery import shared_task

from evilflowers_text.jobs import Outcome, ParagraphsJob, TablesJob


@shared_task(name="evilflowers_text_worker.tables", acks_late=True)
def tables(run_id: str, source: str, output_dir: str, regions: list[dict], **_) -> Outcome:
    """Table regions found by the analyzer's layout pass: markdown, printed caption, crop and position."""
    return TablesJob(run_id, source, output_dir, regions).run()


@shared_task(name="evilflowers_text_worker.paragraphs", acks_late=True)
def paragraphs(run_id: str, source: str, **_) -> Outcome:
    """The paragraph index (IP-018 Phase 2b): paragraphs, sentences, embeddings and links."""
    return ParagraphsJob(run_id, source).run()
