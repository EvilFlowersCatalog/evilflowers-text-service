"""A reader's request, embedded for the catalog's paragraph search (IP-018 Phase 6b).

The catalog waits for the result (30 s by default), so this runs on its own
queue and worker, never behind a book being extracted.
"""

from celery import shared_task

from evilflowers_text.embeddings import Vector, encoder
from evilflowers_text.observability import bind


@shared_task(name="evilflowers_text_worker.embed_query")
def embed_query(text: str, model: str | None = None) -> Vector:
    """Embedded with the model of the run it searches (`run.provenance["paragraphs"]["embedding_model"]`)."""
    # The request text is a reader's query: its length goes on the span and the log lines, never the text.
    with bind(**{"embedding.input_chars": len(text or "")}):
        return encoder(model).query(text)
