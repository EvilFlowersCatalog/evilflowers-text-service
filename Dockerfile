FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OCR, PDF processing, and PyTorch
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libgl1 \
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
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY src/ ./src/

ARG IMAGE_TAG=dev
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    IMAGE_TAG=${IMAGE_TAG} \
    HF_HOME=/models

# Model weights live in a volume, not in the image (IP-018).
VOLUME ["/models"]

# Both queues by default. Compose runs a second container on the embed queue
# alone (`--queues=evilflowers_text_embed`), so a reader's request never waits
# behind a book being indexed.
CMD ["celery", "--workdir", "src", "-A", "main", "worker", "--loglevel=info", "-E", "--pool=solo", "--queues=evilflowers_text_worker,evilflowers_text_embed"]