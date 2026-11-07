.PHONY: install install-all lint format test check worker worker-embed image

# Core + dev tools: enough for the tests, no models.
install:
	uv sync

# Everything the image has (embeddings, OCR/indexing, OpenTelemetry SDK).
install-all:
	uv sync --all-extras

lint:
	uv run ruff check
	uv run ruff format --check

format:
	uv run ruff check --fix
	uv run ruff format

test:
	uv run pytest

check: lint test

# A local worker on both queues (needs `make install-all` and a reachable REDIS_URL).
worker:
	uv run celery -A evilflowers_text.worker worker --loglevel=INFO -E --pool=solo \
		--queues=evilflowers_text_worker,evilflowers_text_embed

worker-embed:
	uv run celery -A evilflowers_text.worker worker --loglevel=INFO -E --pool=solo --queues=evilflowers_text_embed

image:
	docker build --build-arg IMAGE_TAG=local@$$(git rev-parse --short HEAD) -t evilflowers-text-service:local .
