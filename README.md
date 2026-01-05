# EvilFlowers Text Service

A Python microservice that extracts and processes text from PDF documents, with capabilities for extracting structured text content including paragraphs, sentences, and tables.

## Installation

### Install dependencies
```bash
make install
```
or

```bash
pip install -r requirements.txt 
```

## Run the service

### Run with Makefile
```bash
make run
```

### Run directly
```bash
python src/main.py
```

## Project Structure

```
.
├── Dockerfile
├── Makefile
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── main.py
    ├── api.py
    ├── config/
    │   └── Config.py
    └── text_handler/
        ├── TextExtractor.py
        ├── TextProcessor.py
        ├── TableExtractor.py
        └── TextService.py
```

## Architecture Overview

### Core Components

#### 1. TextService (TextHandler)
- Main orchestrator that coordinates text extraction and processing
- Implements singleton pattern
- Manages workflow between extraction and processing components

#### 2. TextExtractor
- Extracts text content from PDF documents using PyMuPDF (fitz)
- Supports multiple extraction levels:
  - Page-level extraction
  - Paragraph extraction
  - Sentence-level extraction
- Validates document paths and handles document loading

#### 3. TextProcessor
- Processes extracted text
- Configurable text processing pipeline

#### 4. TableExtractor
- Specialized component for extracting tabular data from documents
- (Implementation in progress)

### Celery Tasks

The service runs as a Celery worker that processes tasks from a Redis message broker:

- `text_service.process_pdf` (task name)
  - Triggered by EvilFlowersCatalog when a PDF acquisition is created
  - Accepts `acquisition_id` as parameter
  - Downloads PDF from Django API
  - Extracts text content
  - Processes text into chunks
  - Sends processed chunks to search-service for indexing


## Docker Support

Containerized deployment with:
- Python base image
- Celery worker (no HTTP server)
- Temporary file storage for PDF downloads
- Integration with Redis (message broker)
- Integration with search-service for indexing

## Requirements

- Python 3.x
- pip
- make
- Key dependencies:
  - Celery (task queue)
  - Redis (message broker)
  - PyMuPDF (fitz)
  - python-dotenv
  - requests
  - ocrmypdf
  - pytesseract
  - pdf2image
  - unpaper
  - ghostscript

## Integration

The service integrates with:
- **EvilFlowersCatalog** (Django): Receives Celery tasks when PDF acquisitions are created
- **Redis**: Message broker for Celery task queue
- **Search Service**: Sends processed text chunks for indexing
- **Django API**: Downloads PDF files via HTTP API

## Configuration

Environment variables (via .env):
- `REDIS_URL`: Redis broker URL (default: `redis://redis:6379/0`)
- `CATALOG_API_URL`: Django catalog API URL (default: `http://catalog:8000`)
- `CATALOG_API_KEY`: Optional Bearer token for service-to-service authentication (default: `None`)
- `SEARCH_SERVICE_URL`: Search service URL (default: `http://localhost:8001`)
- `CHUNK_SIZE`: Text chunk size for processing (default: `768`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `50`)

**Note**: If `CATALOG_API_KEY` is not set, the service will attempt to download PDFs without authentication. This works for open-access acquisitions. For protected acquisitions, you'll need to configure an API key.