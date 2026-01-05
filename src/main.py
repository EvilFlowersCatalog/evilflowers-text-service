# text_service/main.py
from celery import Celery
import requests
import tempfile
import os
import logging
from text_handler.TextExtractor import TextExtractor
from text_handler.TextProcessor import TextProcessor
from config.Config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

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

@app.task(name='text_service.process_pdf', bind=True)
def process_pdf_task(self, acquisition_url):
    """
    Celery task triggered by Django when Acquisition is created.
    
    Args:
        acquisition_url: Full URL where the acquisition file can be downloaded
    """
    # Extract acquisition_id from URL for logging/document_id purposes
    acquisition_id = acquisition_url.split('/')[-1] if '/' in acquisition_url else acquisition_url
    
    logger.info(f"Starting PDF processing: acquisition_id={acquisition_id}, task_id={self.request.id}")
    
    temp_file_path = None
    try:
        # 1. Download PDF from Django API
        logger.info(f"Downloading PDF from: {acquisition_url}")
        temp_file_path = get_pdf_path(acquisition_url)
        file_size = os.path.getsize(temp_file_path) if temp_file_path else 0
        logger.info(f"PDF downloaded successfully: {file_size:,} bytes")
        
        # 2. Extract text
        logger.info("Extracting text from PDF")
        extractor = TextExtractor(temp_file_path)
        pages, toc = extractor.extract()
        logger.info(f"Text extraction completed: pages={len(pages)}, toc={'yes' if toc else 'no'}")
        
        # 3. Process into chunks
        logger.info("Processing text into chunks")
        processor = TextProcessor()
        processed = processor.process_text(pages, toc, doc_id=str(acquisition_id))
        chunks = processed.get('chunks', [])
        language = processed.get('language', 'unknown')
        logger.info(f"Text processing completed: language={language}, chunks={len(chunks)}")
        
        # 4. Send to search-service
        logger.info(f"Sending {len(chunks)} chunks to search-service for document_id={acquisition_id}")
        response = requests.post(
            f"{Config.SEARCH_SERVICE_URL}/index",
            json={
                "document_id": str(acquisition_id),
                "chunks": {"chunks": chunks}
            },
            timeout=300
        )
        
        if response.status_code != 200:
            logger.error(f"Search indexing failed: status={response.status_code}, response={response.text}")
            raise Exception(f"Search indexing failed: {response.text}")
        
        result = response.json()
        logger.info(f"PDF processing completed successfully: acquisition_id={acquisition_id}, chunks_indexed={result.get('chunks_indexed', 0)}")
        
        return result
        
    except Exception as e:
        logger.error(f"PDF processing failed: acquisition_id={acquisition_id}, error={type(e).__name__}: {str(e)}", exc_info=True)
        raise  # Re-raise to mark task as failed in Celery
        
    finally:
        # Clean up temporary file if created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")

def get_pdf_path(acquisition_url):
    """
    Download PDF from Django API and save to temporary file.
    Returns the path to the temporary PDF file.
    
    Args:
        acquisition_url: Full URL where the acquisition file can be downloaded
    """
    # Prepare headers for service-to-service authentication
    headers = {}
    catalog_secret_key = Config.CATALOG_SECRET_KEY
    if catalog_secret_key:
        headers['X-Internal-Service-Secret'] = catalog_secret_key
    elif Config.CATALOG_API_KEY:
        headers['Authorization'] = f'Bearer {Config.CATALOG_API_KEY}'
    
    response = requests.get(acquisition_url, headers=headers, timeout=300)
    
    if response.status_code != 200:
        logger.error(f"Failed to download PDF: status={response.status_code}, url={acquisition_url}")
        raise Exception(f"Failed to download PDF from catalog API: {response.status_code} - {response.text}")
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file_path = temp_file.name
    
    # Write PDF content to temporary file
    temp_file.write(response.content)
    temp_file.close()
    
    return temp_file_path

if __name__ == '__main__':
    logger.info(f"Text Service Celery Worker starting: broker={Config.REDIS_URL}")
    app.worker_main(['worker', '--loglevel=info', '-E'])  # -E enables task events

