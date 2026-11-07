# EvilFlowers Text Service

The Celery worker that reads the catalog's PDFs. For
[EvilFlowersCatalog](https://github.com/EvilFlowersCatalog/EvilFlowersCatalog) it runs the IP-018 extraction jobs:
tables become markdown with a crop and their position, and the paragraph index records sentences, embeddings and
links. It also embeds readers' queries, and it still chunks every uploaded PDF for `search-service`.

## Tasks

The task names, queues and argument shapes below are a contract with their callers (`tests/test_worker.py` guards
it).

| Task | Queue | Sent by | Does |
|---|---|---|---|
| `evilflowers_text_worker.tables` | `evilflowers_text_worker` | analyzer, kwargs `run_id, source, output_dir, regions` | Reads the analyzer's table regions with pdfplumber into markdown and writes a WebP crop of each. Posts them to `/regions` with `printed_caption` and position. |
| `evilflowers_text_worker.paragraphs` | `evilflowers_text_worker` | analyzer, kwargs `run_id, source, output_dir` | Builds the paragraph index: paragraphs, sentence offsets and quads, section, one 768-d embedding each, and `next`/`similar` links. Posts it to `/paragraphs`. |
| `evilflowers_text_worker.embed_query` | `evilflowers_text_embed` | catalog reader, `args=[text, model]`, waits 30 s | Embeds a reader's request with the model of the run it searches. |
| `evilflowers_text_worker.process_pdf` | `evilflowers_text_worker` | catalog on upload, `args=[source, entry_id]` | Chunks the PDF (with OCR for scans) and indexes it in `search-service`. |
| `evilflowers_text_worker.delete_document` | `evilflowers_text_worker` | no caller today | Removes a document from `search-service`. |
| `evilflowers_text_worker.stats` | `evilflowers_text_worker` | no caller today | Returns `search-service` stats. |

`evilflowers-analyzer-service` dispatches the IP-018 jobs, and each job reports to the catalog's
`/extraction/v1/runs/{run_id}/…` API with `CATALOG_API_KEY`. That key belongs to a catalog user holding
`extraction.ingest_extraction`.

- **Every job reports `done` or `failed`**, because a run finishes only when all of its declared jobs have reported
  (`jobs.ExtractionJob.run`).
- **A 409 is the exception.** It means the catalog no longer accepts the run, and the job then stops quietly.
- **Late acknowledgement.** The jobs acknowledge their messages late, so a worker that dies mid-book has the book
  redelivered.

Paragraph vectors go to the catalog (pgvector), and the run's `provenance.paragraphs.embedding_model` tells the catalog
which model embeds queries. The image consumes both queues. The `text-embed` deployment runs
`--queues=evilflowers_text_embed` alone, so a request never waits behind a book.

## Layout

```
src/evilflowers_text/
├── worker.py          Celery app (`celery -A evilflowers_text.worker worker`), config and routes
├── settings.py        typed settings from the environment (pydantic-settings)
├── schemas.py         messages received and payloads the catalog ingests (mirror of its forms)
├── jobs.py            the IP-018 job protocol (extract → publish → report) and the tables/paragraphs jobs
├── embeddings.py      `Encoder` protocol; sentence-transformers implementation, one model per process
├── tasks/             thin Celery adapters: extraction, embedding, indexing
├── extraction/        paragraphs, tables, page geometry and crops
├── indexing/          search-service pipeline: text extraction + OCR, chunking, whole-document tables
├── clients/           catalog ingestion API, search-service, shared HTTP session
├── observability/     OpenTelemetry setup, spans, metrics, JSON logs, Celery signal wiring
└── cli.py             `evilflowers-text enqueue | download-models`
```

## Development

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. The extras keep the heavy parts out of the default
install:

- `embeddings`: torch and sentence-transformers
- `indexing`: OCR, the layout model and llama-index
- `otel`: the OpenTelemetry SDK

```bash
make install        # core + dev tools (ruff, pytest): enough for the tests, no models
make check          # ruff check, ruff format --check, pytest
make install-all    # everything the image has
make worker         # a local worker on both queues (REDIS_URL from .env)
docker compose up --build                                   # Redis + text + text-embed
docker compose exec text evilflowers-text download-models    # prefetch weights into the models volume
```

Torch is resolved from the PyTorch CPU index (`[tool.uv.sources]`), and the image never pulls CUDA wheels. Bump
dependencies with `uv lock --upgrade-package <name>`.

## Configuration

All settings are environment variables. `.env.sample` lists them, and `src/evilflowers_text/settings.py` has their
defaults and meaning. The ones a deployment sets:

- `REDIS_URL`
- `STORAGE_PATH`, the catalog's storage root, mounted read/write
- `CATALOG_API_URL` and `CATALOG_API_KEY`
- `EMBEDDING_MODEL`, `HF_HOME` (`/models`) and `SEARCH_SERVICE_URL`
- `IMAGE_TAG`, built in by CI

## Observability

The service follows the cluster's telemetry contract (knowledge base, `Cookbooks/Deploy-a-Service` §3).

- **Traces.** Every task gets a CONSUMER span that continues the caller's trace, and every catalog or search-service
  call gets a CLIENT span carrying `traceparent`. Hand-made spans cover PDF parsing, model load and encoding. The
  endpoint comes from `OTEL_EXPORTER_OTLP_ENDPOINT`, or from `HOST_IP` when only that is set; with neither,
  telemetry is off.
- **Metrics** (Mimir):
  - `text_task_duration_seconds`
  - `text_job_outcome_total{job,outcome}`
  - `text_embedding_duration_seconds`, `text_embedding_inputs_total`
  - `text_embedding_model_load_duration_seconds`
  - `text_document_paragraphs`
  - `text_tables_regions_total{markdown}`
  - `text_catalog_retries_total`
  - process CPU and memory
- **Logs.** Each log line is one JSON object on stdout, carrying `level`, `trace_id`/`span_id`, `task_name`/`task_id`,
  and the task's `run_id`, `job`, `entry_id` and `source`.

Queries:

```
Loki:  {service_name=~"text|text-embed"} | json | run_id="<run uuid>"
Tempo: {resource.service.name=~"text|text-embed" && span.evilflowers.extraction.run_id="<run uuid>"}
```
