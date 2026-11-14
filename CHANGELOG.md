# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

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
