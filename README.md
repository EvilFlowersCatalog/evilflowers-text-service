# EvilFlowers Text Service

The Celery worker that reads the catalog's PDFs. For
[EvilFlowersCatalog](https://github.com/EvilFlowersCatalog/EvilFlowersCatalog) it runs the IP-018 extraction jobs:
tables become markdown with a crop and their position, and the paragraph index records sentences, embeddings and
links. Paragraphs are embedded by the shared `tei-embed` (`intfloat/multilingual-e5-large-instruct`, 1024 dims); no
model runs in this process. The catalog embeds readers' queries itself, against the same `tei-embed`.

## Tasks

The task names, queues and argument shapes below are a contract with their callers (`tests/test_worker.py` guards
it).

| Task | Queue | Sent by | Does |
|---|---|---|---|
| `evilflowers_text_worker.tables` | `evilflowers_text_worker` | analyzer, kwargs `run_id, source, output_dir, regions` | Reads the analyzer's table regions with pdfplumber into markdown and writes a WebP crop of each. Posts them to `/regions` with `printed_caption` and position. |
| `evilflowers_text_worker.paragraphs` | `evilflowers_text_worker` | analyzer, kwargs `run_id, source, output_dir` | Builds the paragraph index: paragraphs, sentence offsets and quads, section, one 1024-d embedding each, and `next`/`similar` links. Posts it to `/paragraphs`. |

The search-service tasks (`process_pdf`, `delete_document`, `stats`) and the readers' `embed_query` (queue
`evilflowers_text_embed`) are gone: search moved into the catalog, which calls `tei-embed` directly.

`evilflowers-analyzer-service` dispatches the IP-018 jobs, and each job reports to the catalog's
`/extraction/v1/runs/{run_id}/…` API with `CATALOG_API_KEY`. That key belongs to a catalog user holding
`extraction.ingest_extraction`.

- **Every job reports `done` or `failed`**, because a run finishes only when all of its declared jobs have reported
  (`jobs.ExtractionJob.run`).
- **A 409 is the exception.** It means the catalog no longer accepts the run, and the job then stops quietly.
- **Late acknowledgement.** The jobs acknowledge their messages late, so a worker that dies mid-book has the book
  redelivered.

## Embeddings

`embeddings.TeiEncoder` follows the client contract of the knowledge base (`Proposals/Search-Embeddings`,
`Cookbooks/Shared-Inference`):

- `POST /embed` with `{"inputs": [...], "normalize": true, "truncate": true}`. Paragraphs are documents, sent as plain
  text without an `Instruct:` prefix.
- Batches hold at most 32 inputs, fewer if TEI's `max_client_batch_size` is lower.
- A paragraph longer than `max_input_length` (512 tokens, counted with `POST /tokenize`) is split at sentence starts.
  Its pieces are embedded separately, and the paragraph gets their normalized mean. TEI's truncation is only the
  backstop.
- Each request gets 60 s. 429, 503 and connection errors are retried 3 times with backoff.
- `GET /info` is read once per process. The run's `provenance.paragraphs.embedding_model` is
  `"<model_id>@<model_sha>"`, for example
  `intfloat/multilingual-e5-large-instruct@274baa43b0e13e37fafa6428dbc7938e62e5c439`. The catalog never compares
  vectors of two models.

## Layout

```
src/evilflowers_text/
├── worker.py          Celery app (`celery -A evilflowers_text.worker worker`), config and routes
├── settings.py        typed settings from the environment (pydantic-settings)
├── schemas.py         messages received and payloads the catalog ingests (mirror of its forms)
├── jobs.py            the IP-018 job protocol (extract → publish → report) and the tables/paragraphs jobs
├── embeddings.py      `Encoder` protocol; the tei-embed client (batching, long-paragraph split, retries)
├── tasks/             thin Celery adapters for the extraction jobs
├── extraction/        paragraphs, tables, page geometry and crops
├── clients/           catalog ingestion API, shared HTTP session
└── observability/     OpenTelemetry setup, spans, metrics, JSON logs, Pyroscope profiling, Celery signal wiring
```

## Development

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. The `otel` extra adds the OpenTelemetry SDK, the
`profiling` extra the Pyroscope SDK.

```bash
make install        # core + dev tools (ruff, pytest): enough for the tests
make check          # ruff check, ruff format --check, pytest
make install-all    # everything the image has
make worker         # a local worker (REDIS_URL, TEI_URL, TEI_API_KEY from .env)
docker compose up --build    # Redis + text
```

Bump dependencies with `uv lock --upgrade-package <name>`.

## Configuration

All settings are environment variables. `.env.sample` lists them, and `src/evilflowers_text/settings.py` has their
defaults and meaning. The ones a deployment sets:

- `REDIS_URL`
- `STORAGE_PATH`, the catalog's storage root, mounted read/write
- `CATALOG_API_URL` and `CATALOG_API_KEY`
- `TEI_URL` (default `http://tei-embed.shared.svc.cluster.local`) and `TEI_API_KEY` (1Password
  `ELVIRA/stu.inference.tei/api_key`). From Compose on gateway, also set `TEI_URL=http://192.168.122.227:8081` and
  `TEI_HOST=embed.elvira.internal`.
- `IMAGE_TAG`, built in by CI

The model is whatever `tei-embed` serves. A new model or revision is a new pipeline version in the catalog: run
`backfill_extraction --outdated` there after deploying.

## Observability

The service follows the cluster's telemetry contract (knowledge base, `Cookbooks/Deploy-a-Service` §3).

- **Traces.** Every task gets a CONSUMER span that continues the caller's trace, and every catalog or tei-embed call
  gets a CLIENT span carrying `traceparent`. Hand-made spans cover PDF parsing and each embedding batch. The
  endpoint comes from `OTEL_EXPORTER_OTLP_ENDPOINT`, or from `HOST_IP` when only that is set; with neither,
  telemetry is off.
- **Metrics** (Mimir):
  - `text_task_duration_seconds`
  - `text_job_outcome_total{job,outcome}`
  - `text_embedding_duration_seconds`, `text_embedding_inputs_total`
  - `text_document_paragraphs`
  - `text_tables_regions_total{markdown}`
  - `text_catalog_retries_total`
  - process CPU and memory
- **Logs.** Each log line is one JSON object on stdout, carrying `level`, `trace_id`/`span_id`, `task_name`/`task_id`,
  and the task's `run_id`, `job`, `entry_id` and `source`.
- **Profiles** (Pyroscope). The node's eBPF profiler cannot unwind Python on k3s `control`, so the worker profiles
  itself with the Pyroscope SDK (the `profiling` extra) when `PYROSCOPE_SERVER_ADDRESS` is set:
  `PYROSCOPE_APPLICATION_NAME` (`text`) and `PYROSCOPE_TAGS` (`k=v` pairs separated by commas). It runs in the
  process that runs the tasks, and each trace's root span links to its profile. Query
  `{service_name="text", namespace="evilflowers-dev"}`.

Queries:

```
Loki:  {service_name="text"} | json | run_id="<run uuid>"
Tempo: {resource.service.name="text" && span.evilflowers.extraction.run_id="<run uuid>"}
```
