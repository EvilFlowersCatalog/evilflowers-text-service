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

### API Endpoints

FastAPI server exposing:

- POST `/process_text`
  - Accepts PDF file uploads
  - Extracts text content
  - Stores processed text in Elasticsearch
  - Returns extracted text and document ID

## Configuration

Environment variables (via .env):
- Configurable through Config class
- Uses python-dotenv for environment variable management

## Docker Support

Containerized deployment with:
- Python base image
- FastAPI server exposed on port 8000
- Temporary file storage for uploads
- Integration with Elasticsearch service

## Requirements

- Python 3.x
- pip
- make
- Key dependencies:
  - FastAPI
  - PyMuPDF (fitz)
  - python-dotenv
  - httpx
  - uvicorn
  - ocrmypdf
  - pytesseract
  - pdf2image
  - unpaper
  - ghostscript

## Integration

The service integrates with:
- Elasticsearch service for document storage
- Configurable for additional service integrations