FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OCR, PDF processing, and PyTorch
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    # Tesseract OCR + language data
    tesseract-ocr \
    tesseract-ocr-eng \
    # Add more languages as needed: tesseract-ocr-chi-sim, tesseract-ocr-ara, etc.
    # OCR/PDF processing dependencies
    ghostscript \
    libgs-dev \
    poppler-utils \
    # Image processing for pdf2image
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


ENV PYTHONPATH=/app/src:$PYTHONPATH

# Run the Celery worker
CMD ["celery", "-A", "src.main", "worker", "--loglevel=info", "-E", "--pool=solo", "--queues=evilflowers_text_worker"]