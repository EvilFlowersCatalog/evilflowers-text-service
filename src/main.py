# text_service/main.py
from celery import Celery
import requests
from text_handler.TextExtractor import TextExtractor
from text_handler.TextProcessor import TextProcessor
from config.Config import Config

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
)

@app.task(name='text_service.process_pdf')
def process_pdf_task(acquisition_id):
    """
    Celery task triggered by Django when Acquisition is created
    """
    # 1. Download PDF from MinIO (or get path from Django API)
    pdf_path = get_pdf_path(acquisition_id)
    
    # 2. Extract text
    extractor = TextExtractor(pdf_path)
    text = extractor.extract()
    
    # 3. Process into chunks
    processor = TextProcessor()
    chunks = processor.prepare_chunks(text)
    
    # 4. Send to search-service
    response = requests.post(
        "http://search-service:8001/index",
        json={
            "document_id": str(acquisition_id),
            "chunks": chunks
        },
        timeout=300
    )
    
    if response.status_code != 200:
        raise Exception(f"Search indexing failed: {response.text}")
    
    return response.json()

def get_pdf_path(acquisition_id):
    """Get PDF from Django or MinIO"""
    # Option 1: Call Django API
    # response = requests.get(f"http://catalog:8000/api/acquisitions/{acquisition_id}/")
    # return response.json()['file_path']
    
    # Option 2: Direct MinIO access if you have credentials
    pass

if __name__ == '__main__':
    # For development testing
    print("Text service Celery worker starting...")
    app.worker_main(['worker', '--loglevel=info'])

