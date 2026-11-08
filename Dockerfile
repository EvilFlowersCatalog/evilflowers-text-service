# syntax=docker/dockerfile:1.7

# -- build: the virtual environment, from uv.lock -------------------------------------------------------------------
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first (cached while only the code changes).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --all-extras --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-extras --no-editable

# -- runtime -----------------------------------------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app
COPY --from=build /app/.venv /app/.venv

ARG IMAGE_TAG=dev
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    IMAGE_TAG=${IMAGE_TAG}

# No model in the image or the process: paragraphs are embedded by the shared tei-embed (TEI_URL).
CMD ["celery", "-A", "evilflowers_text.worker", "worker", "--loglevel=INFO", "-E", "--pool=solo", \
     "--queues=evilflowers_text_worker"]
