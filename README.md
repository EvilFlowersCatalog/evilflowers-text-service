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

## Celery tasks

| Task | Queue | Purpose |
|---|---|---|
| `evilflowers_text_worker.process_pdf` | `evilflowers_text_worker` | Chunk a PDF and index it in `search-service` (queued by the catalog on upload) |
| `evilflowers_text_worker.delete_document` | `evilflowers_text_worker` | Remove a document from `search-service` |
| `evilflowers_text_worker.tables` | `evilflowers_text_worker` | IP-018: table regions from the analyzer's layout pass, read with pdfplumber into markdown, cropped to WebP, kept with their position |
| `evilflowers_text_worker.paragraphs` | `evilflowers_text_worker` | IP-018 Phase 2b: the paragraph index (paragraphs, sentence offsets and quads, section, one embedding each, `next` and `similar` links) |
| `evilflowers_text_worker.embed_query` | `evilflowers_text_embed` | IP-018 Phase 6b: embed a reader's request for the catalog with the model of the run it searches |

The IP-018 jobs are dispatched by `evilflowers-analyzer-service` and report to the catalog's
`/extraction/v1/runs/{run_id}/...` ingestion API (`src/catalog.py`) with `CATALOG_API_KEY`, the API key
of a catalog user holding `extraction.ingest_extraction`. Paragraph vectors go to the catalog
(pgvector), not to a vector store of this service. `EMBEDDING_MODEL` defaults to Snowflake Arctic
Embed M v2.0 (768 dimensions, the catalog's column); weights are cached in `HF_HOME` (`/models`).

The image consumes both queues; in compose a second container runs
`--queues=evilflowers_text_embed` alone so requests do not wait behind a book being indexed.

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