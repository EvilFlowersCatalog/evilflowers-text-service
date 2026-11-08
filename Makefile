.PHONY: install install-all lint format test check worker image

# Core + dev tools: enough for the tests.
install:
	uv sync

# Everything the image has (adds the OpenTelemetry SDK).
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

# A local worker (needs a reachable REDIS_URL, and TEI_URL/TEI_API_KEY for the paragraphs job).
worker:
	uv run celery -A evilflowers_text.worker worker --loglevel=INFO -E --pool=solo --queues=evilflowers_text_worker

image:
	docker build --build-arg IMAGE_TAG=local@$$(git rev-parse --short HEAD) -t evilflowers-text-service:local .
