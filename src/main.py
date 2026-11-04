# text_service/main.py
from celery import Celery, Task
import logging
import requests
import os

import fitz

import embeddings
import paragraphs
import settings
import tables
from catalog import Catalog, RunNotRunning
from text_handler.TextService import TextService
from config.Config import Config

logger = logging.getLogger("text")

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
# )
# logger = logging.getLogger(__name__)

# Celery app setup
app = Celery(
    'text_service',
    broker=Config.REDIS_URL, 
    backend=Config.REDIS_URL
)

# Celery config (can be inline instead of separate file)
app.conf.update(
    task_serializer=Config.TASK_SERIALIZER,
    accept_content=Config.ACCEPT_CONTENT,
    result_serializer=Config.RESULT_SERIALIZER,
    timezone='Europe/Bratislava',
    enable_utc=True,
    worker_send_task_events=True,  # Enable task events for monitoring
    task_send_sent_event=True,     # Send events when tasks are sent
    worker_prefetch_multiplier=1,
)

@app.task(bind=True, name="evilflowers_text_worker.stats", queue="evilflowers_text_worker")
def stats_task(self: Task):
    resp = requests.get(f"{Config.SEARCH_SERVICE_URL}/stats", timeout=30)
    resp.raise_for_status()
    stats = resp.json()
    print(stats)
    return stats


@app.task(bind=True, name='evilflowers_text_worker.process_pdf', queue='evilflowers_text_worker')
def process_pdf_task(self: Task, source: str, entry_id: str):
    """
    Celery task triggered by Django when Acquisition is created
    Args:
        source: Relative file path (e.g., "catalogs/mylib/abc123/document.pdf")
        entry_id: The acquisition/entry ID for indexing
    """
    # 1. Convert relative path to absolute path (OCR worker pattern)
    storage_path = os.getenv("STORAGE_PATH", "/mnt/data")
    pdf_path = f"{storage_path}/{source}"
    # Result: "/usr/local/app/private/catalogs/mylib/abc123/document.pdf"

    text_service = TextService(document_path=pdf_path)
    
    # 2. Extract text (file is already on disk via shared volume)
    # Tables are extracted with their positions by the IP-018 `tables` job instead.
    text_results = text_service.extract_text(found_toc=False)

    pages, toc = text_results
    print(toc)

    chunks = text_service.process_text(pages)
    
    # 4. Send to search-service
    print(f"Indexing document ID: {entry_id} with {len(chunks['chunks'])} chunks")
    response = requests.post(
        f"{Config.SEARCH_SERVICE_URL}/index",
        json={
            "document_id": entry_id,
            "chunks": chunks
        },
        timeout=300
    )
    
    # if response.status_code != 200:
    #     logger.error(f"Failed to download PDF: status={response.status_code}, url={acquisition_url}")
    #     raise Exception(f"Failed to download PDF from catalog API: {response.status_code} - {response.text}")
    
    return response.json()


@app.task(bind=True, name="evilflowers_text_worker.delete_document", queue="evilflowers_text_worker")
def delete_document_task(self: Task, document_id: str):
    resp = requests.delete(
        f"{Config.SEARCH_SERVICE_URL}/documents/{document_id}",
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    print(result)
    return result


# if __name__ == '__main__':
#     logger.info(f"Text Service Celery Worker starting: broker={Config.REDIS_URL}")
#     app.worker_main(['worker', '--loglevel=info', '-E'])  # -E enables task events



# IP-018 extraction jobs. The analyzer dispatches them; each reports to the
# catalog's /extraction/v1 ingestion API on its own.


@app.task(name="evilflowers_text_worker.tables", queue=settings.QUEUE)
def tables_task(run_id: str, source: str, output_dir: str, regions: list, **_):
    """Table regions found by the analyzer's layout pass: markdown, caption, crop and position."""
    catalog = Catalog(run_id)
    path = os.path.join(settings.STORAGE_PATH, source)
    try:
        kept = tables.extract(path, output_dir, regions)
    except Exception as error:
        logger.exception("Tables failed for run %s", run_id)
        catalog.job("tables", failed=True, error=f"Cannot read the PDF: {error}")
        return
    try:
        catalog.regions(kept)
        catalog.job("tables", provenance={"image": settings.IMAGE_TAG, "extractor": "pdfplumber"})
    except RunNotRunning:
        logger.warning("Run %s is not accepting results", run_id)


@app.task(name="evilflowers_text_worker.paragraphs", queue=settings.QUEUE)
def paragraphs_task(run_id: str, source: str, **_):
    """The paragraph index (IP-018 Phase 2b): paragraphs, sentences, embeddings and links."""
    catalog = Catalog(run_id)
    path = os.path.join(settings.STORAGE_PATH, source)
    try:
        with fitz.open(path) as document:
            rows = paragraphs.extract(document)
        vectors = embeddings.documents([paragraphs.embedding_text(row) for row in rows]) if rows else []
    except Exception as error:
        logger.exception("Paragraphs failed for run %s", run_id)
        catalog.job("paragraphs", failed=True, error=str(error))
        return

    for row, vector in zip(rows, vectors):
        row["embedding"] = vector
    try:
        catalog.paragraphs(rows, paragraphs.links(vectors, settings.SIMILAR_LINKS_K, settings.SIMILAR_MIN_SCORE))
        catalog.job(
            "paragraphs",
            provenance={"image": settings.IMAGE_TAG, "embedding_model": settings.EMBEDDING_MODEL},
        )
    except RunNotRunning:
        logger.warning("Run %s is not accepting results", run_id)
        return
    except Exception as error:
        # Report it, or the run waits for this job forever.
        logger.exception("Reporting paragraphs failed for run %s", run_id)
        try:
            catalog.job("paragraphs", failed=True, error=str(error))
        except Exception:
            logger.exception("Could not report the failed paragraphs job for run %s", run_id)
        return
    logger.info("Run %s: %d paragraphs", run_id, len(rows))


@app.task(name="evilflowers_text_worker.embed_query", queue=settings.EMBED_QUEUE)
def embed_query_task(text: str, model: str = None):
    """A reader's request, embedded with the model of the run it searches (catalog Phase 6b)."""
    return embeddings.query(text, model or settings.EMBEDDING_MODEL)
