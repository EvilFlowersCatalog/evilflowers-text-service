# text_service/main.py
from celery import Celery, Task
import requests
import os
from text_handler.TextService import TextService
from config.Config import Config

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
    text_results = text_service.extract_text(found_toc=False)
    tables = text_service.extract_tables()

    pages, toc = text_results

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

    print(response)
    
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

