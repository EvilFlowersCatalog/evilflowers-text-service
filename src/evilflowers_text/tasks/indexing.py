"""search-service indexing, queued by the catalog on every PDF upload (IP-018 keeps it, Q6)."""

import logging
from typing import Any

from celery import shared_task

from evilflowers_text.clients import SearchServiceClient
from evilflowers_text.indexing.pipeline import IndexingPipeline
from evilflowers_text.observability import bind
from evilflowers_text.settings import get_settings

logger = logging.getLogger(__name__)


@shared_task(name="evilflowers_text_worker.process_pdf")
def process_pdf(source: str, entry_id: str) -> Any:
    """Chunk a PDF and index it in search-service.

    Args:
        source: path relative to the storage root (``catalogs/<slug>/<entry>/<file>.pdf``).
        entry_id: the catalog entry; search-service's document id.
    """
    with bind(entry_id=entry_id, source=source):
        chunks = IndexingPipeline().chunks(get_settings().source_path(source), entry_id)
        logger.info("Indexing document %s with %d chunks", entry_id, chunks["total_chunks"])
        return SearchServiceClient().index(entry_id, chunks)


@shared_task(name="evilflowers_text_worker.delete_document")
def delete_document(document_id: str) -> Any:
    with bind(entry_id=document_id):
        result = SearchServiceClient().delete(document_id)
        logger.info("Deleted document %s from search-service: %s", document_id, result)
        return result


@shared_task(name="evilflowers_text_worker.stats")
def stats() -> Any:
    result = SearchServiceClient().stats()
    logger.info("Search-service stats: %s", result)
    return result
