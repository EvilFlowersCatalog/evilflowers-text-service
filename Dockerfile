FROM python:3.12-slim

WORKDIR /app

# Install system dependencies FIRST (needed for PyTorch, sentence-transformers, etc)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose port
EXPOSE 8000

# Run the FastAPI application
CMD ["celery", "-A", "src.main", "worker", "--loglevel=info"]
