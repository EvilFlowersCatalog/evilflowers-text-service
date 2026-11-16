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
| `evilflowers_text_worker.tables` | `evilflowers.text.tables` (`TABLES_QUEUE`) | analyzer, kwargs `run_id, source, output_dir, regions`, optional `attempt`, `job` (a batch: `tables:0001-0008`) | Reads the analyzer's table regions with pdfplumber into markdown and writes a WebP crop of each. Posts them to `/regions` with `printed_caption` and position. |
| `evilflowers_text_worker.paragraphs` | `evilflowers.text.paragraphs` (`PARAGRAPHS_QUEUE`) | analyzer, kwargs `run_id, source, output_dir`, optional `attempt`, `outline` | Builds the paragraph index: paragraphs, sentence offsets and quads, section, one 1024-d embedding each, and `next`/`similar` links. Posts it to `/paragraphs`. |

The kwargs are the shared contract's `RegionsJob` and `ParagraphsJob`
([evilflowers-worker](https://github.com/EvilFlowersCatalog/evilflowers-worker) `contract.py`). Unknown kwargs are
ignored. A message that does not validate is reported `failed` and never retried.

Each job has its own queue (issue #13): tables wait on the table review, paragraphs on the bulk embed lane, and in
one shared queue a slow table review held up every paragraph task behind it. Two lists also let each job be scaled
on its own. The analyzer names the queue when it sends. The worker consumes the queues in `TEXT_QUEUE`
(comma-separated). Left empty, it consumes both, plus the legacy `evilflowers_text_worker`, so messages already
there or sent by an older analyzer are still run. A `--queues` option on the command line overrides `TEXT_QUEUE`.

The search-service tasks (`process_pdf`, `delete_document`, `stats`) and the readers' `embed_query` (queue
`evilflowers_text_embed`) are gone: search moved into the catalog, which calls `tei-embed` directly.

`evilflowers-analyzer-service` dispatches the IP-018 jobs, and each job reports to the catalog's
`/extraction/v1/runs/{run_id}/…` API with `CATALOG_API_KEY`. That key belongs to a catalog user holding
`extraction.ingest_extraction`.

The runtime is the shared [evilflowers-worker](https://github.com/EvilFlowersCatalog/evilflowers-worker) kit (issue
#16), pinned to a commit in `pyproject.toml`. It owns the job protocol, the catalog client, the contract, the Celery
app, the delivery cap, the scale-in shutdown and the telemetry. Its `docs/architecture.md` is the design. This
repository holds only the domain: table reading, paragraph segmentation, the TEI client and the table review.

- **Every job reports `done` or `failed`**, because a run finishes only when all of its declared jobs have reported
  (`evilflowers_worker.jobs.LeafJob.run`).
- **Each job reports under its own name.** That is the message's `job` when it names a batch (`tables:0001-0008`,
  when the analyzer runs with `DISPATCH_PER_BATCH`), and the kind (`tables`, `paragraphs`) otherwise.
- **Each job works for one attempt** (#14, EvilFlowersCatalog#140). The analyzer passes the run's `attempt` from
  `/start`, and every write carries it as `?attempt=`. The catalog refuses the writes of a superseded attempt with
  409 `superseded`. The job then reports `superseded` (the catalog counts these) and is never retried.
- **A 409 `not_running` stops the job quietly**: the catalog no longer accepts the run. A 409 `duplicate` (the run
  already holds these paragraphs) is reported `failed`, since the run still waits for the job.
- **Copies stop before the work** (issue #12). A job can arrive twice for one run: the analyzer's own task may be
  redelivered and dispatch every job again, or a message comes back after a worker died. Before reading the PDF, a
  job reads its run (`GET /extraction/v1/runs/{run_id}`). It goes on only while the run is `running` at its
  attempt and its own job is `pending`. Otherwise it records outcome `not_running`, `already_done` or `superseded`,
  and posts nothing. While a book is embedded, the job reads the run again every `RUN_RECHECK_INTERVAL` seconds
  (default 60), so a copy superseded mid-book stops within a minute instead of embedding the whole book.
- **Late acknowledgement and a delivery cap.** The jobs acknowledge their messages late, and a message whose worker
  process dies (an OOM, or the whole container) is requeued. Every delivery is counted in Redis. Past
  `MAX_DELIVERIES` (default 5) the job is reported `failed` instead of run again, so a book that always kills the
  worker neither loops nor leaves its run waiting.
- **Scale-in requeues the running job** (#17). The image runs `--pool=prefork --concurrency=1`, still one job at a
  time; solo cannot cancel a task. SIGTERM is Celery's warm shutdown: a job that finishes is acknowledged, and an
  idle worker exits at once. After `WORKER_SOFT_SHUTDOWN_TIMEOUT` (default 10 s) the worker cancels the job, and
  Redis has its message back at once. If the cancel has not stopped it after `WORKER_CANCEL_TIMEOUT` (30 s), the
  worker restores the message itself and exits. A deliberate requeue does not count towards the delivery cap. Do
  not set `REMAP_SIGTERM`. A job cancelled in the middle of posting paragraphs leaves part of them stored. Its copy
  then gets 409 `duplicate` and is reported `failed`.

### Sections

A paragraph's `section` is the innermost outline heading in force on its page (#15). The outline is the analyzer's
`outline` kwarg when it sends one: the bookmarks, or headings it detected on the text layer of a book without any.
The kwarg is authoritative even when it is `[]`. Without it (an older analyzer), the PDF's bookmarks are read. The
run's `provenance.paragraphs.outline` says which: `analyzer` or `bookmarks`.

## Table decision

With `DECIDER_ENABLED` and `DECIDER_VISION_URL` set, each table crop goes to `decider-vision` first, through the
kit's judges (evilflowers-worker#6, the `judges` extra). The question is `decider.IS_TABLE`, a yes/no (`noul`) asking
"is this a data table, not a table of contents, a form or page layout?", rejected on `no`. The kit owns the HTTP
client, its circuit breaker (`JUDGE_BREAKER_*`), the per-job budget (`JUDGE_BUDGET_*`), the band of the answer
(`JUDGE_THRESHOLDS`) and `text_judge_total{question="is_table", outcome, band}`. Sequence:
`docs/diagrams/sequence-table-decision.puml`.

- The kit bands the answer `hide` (a `no` at or above `JUDGE_THRESHOLDS.is_table.hide`): the Ollama review is skipped
  and the region is posted with `band: hide`, `p`, `decision` and no `score`. A table of contents is fully grounded in
  the PDF text, so a score would only band it upward.
- Otherwise the review runs as before, then the **table score** is the weighted mean of what is known: grounded share
  ×3, grid quality (filled share × (1 − fragment share), 0 for an implausible grid) ×2, review outcome ×1 (`kept` 1,
  `rewritten` and `transcribed` 0.5, `unusable` and `failed` left out), and ×1 the probability that it is a data
  table. With `DECIDER_TABLE_BANDS=hide_below,keep_from` the band is `hide`, `review` or `keep`. Without it no band is
  sent, and the catalog bands from `score` once its own thresholds are set.
- The region carries the kit's `band`, `p` (of the answer given), `decision` (`yes` / `no`) and `decider_model`
  (`model@revision`, with `DECIDER_VISION_REVISION`), plus `score`, `review_outcome`, `grounded_share` and `signals`
  (`filled_share`, `fragment_share`, `strategy`, `rotated_share`), the fields of EvilFlowersCatalog#189. The issue-#19
  keys stay in `provenance`. The crop is kept in every band.
- Set `JUDGE_THRESHOLDS` for `is_table` before any score bands (here or the catalog's). The decider is 1/7 of the
  score, so a well-grounded table of contents scores about 0.86 even when it is surely not a table. Only the skip gate
  hides non-tables.
- No answer (judges off, decider-vision unreachable or its breaker open, the job's budget spent, an answer of an
  unknown shape): none of these fields is sent, and the region is posted exactly as before #20. The kit logs
  `Judge decider-vision unreachable` (or `refused` / `invalid`) and counts the outcome.
- `text_table_decision_total{band}` counts the bands of answered tables: `keep`, `review`, `hide`, `none`.

## Embeddings

`embeddings.TeiEncoder` follows the client contract of the knowledge base (`Proposals/Search-Embeddings`,
`Cookbooks/Shared-Inference`):

- `POST /embed` with `{"inputs": [...], "normalize": true, "truncate": true}`. Paragraphs are documents, sent as plain
  text without an `Instruct:` prefix.
- Batches hold at most 32 inputs, fewer if TEI's `max_client_batch_size` is lower.
- A paragraph longer than `max_input_length` (512 tokens, counted with `POST /tokenize`) is split at sentence starts.
  Its pieces are embedded separately, and the paragraph gets their normalized mean. TEI's truncation is only the
  backstop.
- Each request gets 60 s. 429, 503 and connection errors are retried 3 times with backoff, per batch.
- Up to `TEI_CONCURRENCY` batches are in flight at once (default 1: every `/tokenize`, then every `/embed`, one at a
  time). Above 1, a text batch's pieces are embedded as soon as its `/tokenize` is back, and the router spreads the
  requests over the lane's replicas. The `/embed` batches and the vectors' order are the same at any concurrency.
- `TEI_LANE` (`bulk` or `interactive`), when set, goes out as the `x-embed-lane` header on every request and picks the
  router's lane, over the Host header (knowledge base, `Proposals/Embed-Lanes`). This worker sends `bulk`.
- `GET /info` is read once per process. The run's `provenance.paragraphs.embedding_model` is
  `"<model_id>@<model_sha>"`, for example
  `intfloat/multilingual-e5-large-instruct@274baa43b0e13e37fafa6428dbc7938e62e5c439`. The catalog never compares
  vectors of two models.

## Layout

```
src/evilflowers_text/
├── worker.py          Celery app (`celery -A evilflowers_text.worker worker`): the kit's create_app, queues, routes
├── settings.py        the kit's WorkerSettings plus queues, table review, TEI and paragraph settings
├── jobs.py            TablesJob and ParagraphsJob: the kit's LeafJob (extract, publish, provenance)
├── embeddings.py      `Encoder` protocol; the tei-embed client (batching, long-paragraph split, retries, concurrency)
├── metrics.py         text's own instruments (embeddings, paragraphs, tables), beside the kit's standard set
├── tasks/             the task bodies: `runtime.run_job(...)`
└── extraction/        paragraphs, tables, table review, page geometry and crops
```

Tests use the kit's doubles: `FakeCatalog` in process, and `StubCatalogServer` + `WorkerProcess` for one real
prefork worker of the production app against Redis (`tests/test_end_to_end.py`). That test needs `TEST_REDIS_URL`
(default `redis://localhost:6379/15`). Without Redis it is skipped, and in CI it fails.

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
- `TEXT_QUEUE`, only for a Deployment that runs one job: `evilflowers.text.tables` or `evilflowers.text.paragraphs`
- `WORKER_SOFT_SHUTDOWN_TIMEOUT` (10–30 s) and `WORKER_CANCEL_TIMEOUT` (30 s): their sum below the pod's
  `terminationGracePeriodSeconds`. No `REMAP_SIGTERM`.
- `STORAGE_PATH`, the catalog's storage root, mounted read/write
- `CATALOG_API_URL` and `CATALOG_API_KEY`
- `TEI_URL` (default `http://tei-embed.shared.svc.cluster.local`) and `TEI_API_KEY` (1Password
  `ELVIRA/stu.inference.tei/api_key`). From Compose on gateway, also set `TEI_URL=http://192.168.122.227:8081` and
  `TEI_HOST=embed.elvira.internal`.
- `TEI_LANE=bulk`, and `TEI_CONCURRENCY` (3–4 on dev). It bounds the requests in flight per worker process, so the
  bulk lane (1–3 `tei-embed-bulk` replicas by HPA) sees up to worker replicas × `TEI_CONCURRENCY`.
- `DECIDER_ENABLED`, `DECIDER_VISION_URL` (`http://decider-vision.shared.svc.cluster.local`) and `DECIDER_TIMEOUT`
  on `text-tables` (#20), and `DECIDER_VISION_REVISION` (the weights' snapshot). `JUDGE_THRESHOLDS` (`is_table`) and
  `DECIDER_TABLE_BANDS` stay empty until knowledge#65 calibrates them. See [Table decision](#table-decision).
- `IMAGE_TAG`, built in by CI

The model is whatever `tei-embed` serves. A new model or revision is a new pipeline version in the catalog: run
`backfill_extraction --outdated` there after deploying.

## Observability

The service follows the cluster's telemetry contract (knowledge base, `Cookbooks/Deploy-a-Service` §3).

- **Traces.** Every task gets a CONSUMER span that continues the caller's trace, and every catalog or tei-embed call
  gets a CLIENT span carrying `traceparent`, except the tei-embed batches. Hand-made spans cover PDF parsing and
  each `embeddings.documents` call, with its batch count, retries, slowest batch, `tei.concurrency` and the
  `tei.lane` that served it (issue #10: no per-batch spans). The
  endpoint comes from `OTEL_EXPORTER_OTLP_ENDPOINT`, or from `HOST_IP` when only that is set; with neither,
  telemetry is off.
- **Metrics** (Mimir):
  - `text_task_duration_seconds`
  - `text_job_outcome_total{job,outcome}` (`done`, `failed`, `not_running`, `already_done`, `superseded`; `job` is
    the kind, never a batch name)
  - `text_deliveries_capped_total`
  - `text_embedding_duration_seconds{kind,model,lane}`, `text_embedding_inputs_total{kind,model,lane}` (`lane` when
    the router answers `x-embed-lane`)
  - `text_document_paragraphs`
  - `text_tables_regions_total{markdown}`
  - `text_catalog_retries_total{route,reason}`
  - process CPU and memory
- **Logs.** Each log line is one JSON object on stdout, carrying `level`, `trace_id`/`span_id`, `task_name`/`task_id`,
  and the job's `run_id`, `job`, `attempt`, `source`, `task_id` and `delivery`.
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
