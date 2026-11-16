# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- The table review is measured. `TableReviewer.review` returns what it did: `kept`, `rewritten`, `transcribed`,
  `unusable` or `failed`. The new counter `text_table_review_total{outcome}` counts each outcome, and a
  `Table read` log line per table gives its page, outcome, grounded share and grid strategy, with the job's `run_id`.
  Each table region's `provenance` now carries `review_outcome`, `grounded_share`, `grid_strategy`
  (`lines` | `text` | null), `grid_filled_share`, `grid_fragment_share` and `rotated_share`. `grounded_share` is the
  share of the markdown's numbers and words found in the PDF text inside the table's box. It is null when there is no
  markdown, no text layer or rotated text. Words match loosely, across line breaks, so the share moves mainly with the
  numbers. The catalog stores the per-region provenance as it is today, and
  [EvilFlowersCatalog#189](https://github.com/EvilFlowersCatalog/EvilFlowersCatalog/issues/189) will promote these
  fields to columns. ([#19](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/19))

### Changed

- The runtime is now the shared [evilflowers-worker](https://github.com/EvilFlowersCatalog/evilflowers-worker) kit,
  pinned to `c37d860`. Its job protocol (`LeafJob`), catalog client, contract models, Celery app (`create_app`),
  delivery cap, shutdown and telemetry replace the copies in this repository: `observability/`, `clients/`,
  `schemas.py`, `jobs.ExtractionJob`, `retry_on_worker_lost` / `report_lost_job` and `requeue_on_sigterm` are gone.
  Both task bodies are `runtime.run_job(...)`. A message that does not validate is reported `failed` and never
  retried. ([#16](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/16))
- Jobs take `attempt` and `job`. Every write carries `?attempt=`. A job reads its run before the work and again every
  `RUN_RECHECK_INTERVAL` (60 s) while a book is embedded, and stops as `superseded` when a later `/start` has begun a
  new attempt. A `tables` task with `job="tables:0001-0008"` reports under that batch's name, and without `job` under
  `tables`, as before. This lets the analyzer turn `DISPATCH_PER_BATCH` on once image and equation report per batch
  too. ([#14](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/14),
  [#16](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/16))
- Paragraph sections come from the analyzer's `outline` kwarg when it is present, even when it is `[]`, and from the
  PDF's bookmarks otherwise. Books without bookmarks now get sections from the headings the analyzer detects.
  `provenance.paragraphs.outline` records which source was used (`analyzer` | `bookmarks`).
  ([#15](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/15))
- Scale-in is the kit's `evilflowers_worker.shutdown`. SIGTERM stays a warm shutdown, so a job that finishes in time
  is acknowledged and not redone. After `WORKER_SOFT_SHUTDOWN_TIMEOUT` (default still 10 s) the job is cancelled and
  requeued. If the kill is denied, the worker restores the message itself after `WORKER_CANCEL_TIMEOUT` (30 s) and
  exits. `WORKER_ENABLE_SOFT_SHUTDOWN_ON_IDLE` is no longer read, and `REMAP_SIGTERM` must not be set.
  ([#17](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/17))
- A job whose worker process dies is requeued as many times as `MAX_DELIVERIES` (default 5) allows, counted in Redis
  across container restarts, and then reported `failed`. Before, the limit was 2 Celery retries 60 s apart, and it
  only covered a killed child. ([#16](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/16))
- Settings inherited from the kit: `REDIS_URL` now defaults to db 7 (was db 0; every deployment sets it),
  `BROKER_VISIBILITY_TIMEOUT` must be at least 3600, and `CATALOG_BATCH_SIZE` at most 1000. A 409 `duplicate` from
  the catalog is now reported `failed` instead of being dropped as `not_running`.
- Telemetry: `text_job_outcome_total` gains outcome `superseded`, and `text_deliveries_capped_total` is new.
  `text_catalog_retries_total` is now labelled `route` (`/jobs`, `/paragraphs` …) instead of `path`. Log lines
  carry `attempt`, `task_id` and `delivery`.

- Each job has its own queue: `tables` → `evilflowers.text.tables` (`TABLES_QUEUE`), `paragraphs` →
  `evilflowers.text.paragraphs` (`PARAGRAPHS_QUEUE`), each on a direct exchange of the same name.
  The names follow the `evilflowers.<service>` scheme of
  [knowledge#14](https://github.com/EvilFlowersCatalog/knowledge/issues/14).
  A slow table review no longer holds up paragraph tasks, and each list can be scaled on its own.
  `TEXT_QUEUE` is now a comma-separated list of the queues to consume. Left empty (the new
  default), it means both, plus the legacy `evilflowers_text_worker`, so messages already there or
  from an older analyzer still run. The analyzer has to send to the new queues, and a `--queues`
  option in the k8s args overrides all of this.
  ([#13](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/13))
- SIGTERM puts the running job's message back on its queue after `WORKER_SOFT_SHUTDOWN_TIMEOUT`
  (default 10 s) instead of after the visibility timeout (12 h). This is Celery's soft shutdown
  (`WORKER_ENABLE_SOFT_SHUTDOWN_ON_IDLE` is also read), with SIGTERM made a cold shutdown. It
  needs a pool that can cancel a task, so the image now runs `--pool=prefork --concurrency=1`
  (still one job at a time). Under `--pool=solo`, SIGTERM stays a warm shutdown: solo cannot
  cancel a task, and the failed cancel would get the message acknowledged and lost.
  ([#13](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/13))
- Under prefork, a job whose child process dies (an OOM) is sent again after 60 s, up to 2
  times, and then reported `failed`. Celery's default would acknowledge the message silently, and
  the run would wait for that job forever.
  ([#13](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/13))
- Before reading the PDF, a job reads its run (`GET /extraction/v1/runs/{id}`). It stops, posting
  nothing, unless the run is `running` and its job `pending`. The new outcome `already_done` and
  the existing `not_running` count these copies. A second copy of `paragraphs` for a run whose
  paragraphs are stored used to embed the whole book again (hours on tei-embed) only to get a 409
  at the end. ([#12](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/12))

- The default `BROKER_VISIBILITY_TIMEOUT` is now 43200 s (12 h), up from 7200 s. Every consumer
  of a Redis db restores all unacknowledged messages older than its own timeout, in any queue. On
  dev, `text` shares db 7 with the catalog, analyzer, image, equation and lcpencrypt, and an equation
  pass has taken over 2 h, so 2 h caused redeliveries. 12 h matches the catalog and the analyzer.
  A task lost to a killed pod now comes back after up to 12 h instead of 2 h. The variable still
  overrides the default. ([#9](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/9))
