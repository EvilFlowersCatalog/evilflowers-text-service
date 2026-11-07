"""OpenTelemetry for the text worker: traces, metrics and JSON logs.

The contract is the cluster's (knowledge base, Cookbooks/Deploy-a-Service §3):

- configuration is the standard ``OTEL_*`` environment. The pod sets ``HOST_IP``
  (downward API) and ``OTEL_EXPORTER_OTLP_ENDPOINT=http://$(HOST_IP):4318``;
  when only ``HOST_IP`` is set, the endpoint is derived from it. With neither,
  telemetry stays off and the worker behaves as if this module did not exist;
- traces and metrics go over OTLP/HTTP to the node's Alloy agent;
- logs are NOT exported over OTLP: Alloy tails stdout, so every line is one JSON
  object with ``level`` and, inside a span, ``trace_id`` (Loki → Tempo link).

``connect()`` wires everything to Celery signals. ``setup()`` runs on
``worker_process_init``, which fires in the process that runs the tasks for both
pools: the solo pool sends it from the worker itself, the prefork pool from each
child after the fork (exporter threads do not survive a fork).

What is traced:

- every task (CONSUMER span; continues the producer's trace when the caller,
  catalog or analyzer, propagates the context in the task headers);
- every ``requests`` call (catalog ingestion API, search-service): CLIENT spans
  carrying ``traceparent``, so the catalog's Django span joins the same trace;
- hand-made spans where the time goes (PDF parsing, model load, encoding).

Redis is deliberately not instrumented: the Celery broker polls it constantly
and every poll would become a root trace.

OpenTelemetry is imported lazily or guarded, so the unit tests (and an image
built without it) run unchanged.
"""

import json
import logging
import os
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Optional

import settings

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:  # pragma: no cover - the image always ships the API
    metrics = None
    trace = None

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "evilflowers-text"
SCOPE = "evilflowers.text"
OTLP_HTTP_PORT = 4318

ENVIRONMENT_DEFAULTS = {
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    "OTEL_SEMCONV_STABILITY_OPT_IN": "http",
    # The k8s attributes (pod, node, namespace) are added by Alloy's k8sattributes processor.
    "OTEL_EXPERIMENTAL_RESOURCE_DETECTORS": "process,os,host",
}

# Process metrics only: container CPU/memory come from cAdvisor, profiles from
# Pyroscope. The resident memory shows what a loaded embedding model costs.
SYSTEM_METRICS_CONFIG = {
    "process.cpu.time": ["user", "system"],
    "process.memory.usage": None,
    "process.thread.count": None,
    "cpython.gc.collections": None,
}

# Seconds. Covers a 20 ms query embedding up to a 30 min book.
DURATION_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800]
COUNT_BUCKETS = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

_lock = threading.Lock()
_state = {"done": False, "enabled": False}
_providers = []
_instruments: Dict[str, object] = {}
_task_started: Dict[str, float] = {}


# -- setup -------------------------------------------------------------------


def endpoint() -> Optional[str]:
    """The OTLP endpoint: explicit ``OTEL_EXPORTER_OTLP_*ENDPOINT``, else derived from ``HOST_IP``."""
    for name in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    host = os.environ.get("HOST_IP", "").strip()
    if not host:
        return None
    if ":" in host:
        host = f"[{host}]"
    derived = f"http://{host}:{OTLP_HTTP_PORT}"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = derived
    return derived


def is_enabled() -> bool:
    return bool(_state["enabled"])


def setup() -> bool:
    """Configure tracing and metrics for this process. Idempotent; returns whether telemetry is on."""
    with _lock:
        if _state["done"]:
            return _state["enabled"]
        _state["done"] = True

        if trace is None or os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
            return False
        target = endpoint()
        if target is None:
            logger.info("OpenTelemetry off: no OTLP endpoint and no HOST_IP")
            return False
        try:
            _configure()
        except ImportError as error:
            logger.warning("OpenTelemetry SDK is not installed (%s); telemetry is off", error)
            return False
        _state["enabled"] = True
        logger.info("OpenTelemetry on: endpoint=%s service=%s", target, os.environ.get("OTEL_SERVICE_NAME"))
        return True


def _configure() -> None:
    for name, value in ENVIRONMENT_DEFAULTS.items():
        os.environ.setdefault(name, value)

    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = build_resource()

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    # Exemplars: histogram points recorded inside a sampled span carry its trace id.
    meter_provider = MeterProvider(
        resource=resource, metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())]
    )
    metrics.set_meter_provider(meter_provider)
    _providers[:] = [tracer_provider, meter_provider]

    CeleryInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    SystemMetricsInstrumentor(config=SYSTEM_METRICS_CONFIG).instrument()


def build_resource():
    """Detectors, then the environment, then our defaults for what the environment leaves out.

    ``service.instance.id`` is ``<hostname>-<pid>`` (the pod name in k8s), so a
    restarted pod is a new series and an operator can match it against ``ps``.
    """
    from opentelemetry.sdk.resources import OTELResourceDetector, Resource

    from_environment = OTELResourceDetector().detect().attributes
    defaults = {
        "service.name": os.environ.get("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME,
        "service.namespace": "elvira",
        "service.version": settings.IMAGE_TAG,
        "service.instance.id": f"{socket.gethostname()}-{os.getpid()}",
        "evilflowers.text.embedding_model": settings.EMBEDDING_MODEL,
    }
    return Resource.create({key: value for key, value in defaults.items() if key not in from_environment})


def shutdown(*_args, **_kwargs) -> None:
    """Flush and stop the providers. Idempotent; never fails the exit path."""
    with _lock:
        providers, _providers[:] = list(_providers), []
    for provider in providers:
        try:
            provider.shutdown()
        except Exception:  # pragma: no cover
            logger.debug("OpenTelemetry provider shutdown failed", exc_info=True)


# -- spans -------------------------------------------------------------------


def _clean(attributes: dict) -> dict:
    return {key: value for key, value in attributes.items() if value is not None}


@contextmanager
def span(name: str, **attributes):
    """A child span of the current one; a no-op without OpenTelemetry. Exceptions mark it failed."""
    if trace is None:
        yield None
        return
    with trace.get_tracer(SCOPE).start_as_current_span(name, attributes=_clean(attributes)) as current:
        yield current


def annotate(**attributes) -> None:
    """Attributes on the current span (the task's own span when called first thing in a task)."""
    if trace is None:
        return
    current = trace.get_current_span()
    for key, value in _clean(attributes).items():
        current.set_attribute(key, value)


def fail(error: BaseException) -> None:
    """Mark the current span failed for an exception the code catches (and reports) itself."""
    if trace is None:
        return
    current = trace.get_current_span()
    current.record_exception(error)
    current.set_status(Status(StatusCode.ERROR, str(error)[:200]))


# -- metrics -----------------------------------------------------------------
# Low-cardinality attributes only: never a run id or document id (those go on spans).


def _instrument(name: str):
    if metrics is None:
        return None
    with _lock:
        if name not in _instruments:
            meter = metrics.get_meter(SCOPE)  # a proxy: becomes real once setup() sets the provider
            factory = {
                "task.duration": lambda: meter.create_histogram(
                    "text.task.duration", "s", "Celery task run time",
                    explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
                ),
                "job.outcome": lambda: meter.create_counter(
                    "text.job.outcome", "{job}", "IP-018 jobs reported to the catalog, by outcome",
                ),
                "embedding.duration": lambda: meter.create_histogram(
                    "text.embedding.duration", "s", "Time to embed one batch (documents) or one query",
                    explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
                ),
                "embedding.inputs": lambda: meter.create_counter(
                    "text.embedding.inputs", "{text}", "Texts embedded",
                ),
                "model.load": lambda: meter.create_histogram(
                    "text.embedding.model_load.duration", "s", "Embedding model load time (once per process)",
                    explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
                ),
                "paragraphs": lambda: meter.create_histogram(
                    "text.document.paragraphs", "{paragraph}", "Paragraphs indexed per document",
                    explicit_bucket_boundaries_advisory=COUNT_BUCKETS,
                ),
                "catalog.retries": lambda: meter.create_counter(
                    "text.catalog.retries", "{retry}", "Retried calls to the catalog ingestion API",
                ),
            }[name]
            _instruments[name] = factory()
        return _instruments[name]


def record_task(task_name: str, state: str, seconds: float) -> None:
    histogram = _instrument("task.duration")
    if histogram is not None:
        histogram.record(seconds, {"celery.task.name": task_name, "celery.task.state": state or "UNKNOWN"})


def record_job(job: str, outcome: str) -> None:
    """``outcome``: done, failed or not_running (the catalog refused the results)."""
    counter = _instrument("job.outcome")
    if counter is not None:
        counter.add(1, {"job": job, "outcome": outcome})


def record_embedding(kind: str, model: str, inputs: int, seconds: float) -> None:
    """``kind``: documents (paragraph batches) or query (a reader's request)."""
    attributes = {"kind": kind, "model": model}
    histogram, counter = _instrument("embedding.duration"), _instrument("embedding.inputs")
    if histogram is not None:
        histogram.record(seconds, attributes)
    if counter is not None:
        counter.add(inputs, attributes)


def record_model_load(model: str, seconds: float) -> None:
    histogram = _instrument("model.load")
    if histogram is not None:
        histogram.record(seconds, {"model": model})


def record_paragraphs(count: int) -> None:
    histogram = _instrument("paragraphs")
    if histogram is not None:
        histogram.record(count)


def record_catalog_retry(path: str, reason: str) -> None:
    counter = _instrument("catalog.retries")
    if counter is not None:
        counter.add(1, {"path": path, "reason": reason})


# -- logs --------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ``time``, ``level``, ``logger``, ``message``, then the
    trace and Celery task when there is one, and the traceback."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if trace is not None:
            context = trace.get_current_span().get_span_context()
            if context.is_valid:
                payload["trace_id"] = format(context.trace_id, "032x")
                payload["span_id"] = format(context.span_id, "016x")
        task = _current_task()
        if task is not None:
            payload["task_name"], payload["task_id"] = task
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _current_task():
    try:
        from celery import current_task

        if current_task and current_task.request.id:
            return current_task.name, current_task.request.id
    except Exception:  # pragma: no cover - outside a worker
        pass
    return None


def use_json_logs(logger: logging.Logger = None, **_kwargs) -> None:
    """Celery's ``after_setup_logger`` / ``after_setup_task_logger`` receiver."""
    if settings.LOG_FORMAT != "json":
        return
    target = logger or logging.getLogger()
    for handler in target.handlers:
        handler.setFormatter(JsonFormatter())


# -- Celery wiring -------------------------------------------------------------


def _task_prerun(task_id=None, **_kwargs):
    if task_id:
        _task_started[task_id] = time.perf_counter()


def _task_postrun(task_id=None, task=None, state=None, **_kwargs):
    started = _task_started.pop(task_id, None)
    if started is not None and task is not None:
        record_task(task.name, state, time.perf_counter() - started)


def connect() -> None:
    """Hook telemetry and JSON logs into the Celery worker (call once, where the app is defined)."""
    from celery import signals

    signals.worker_process_init.connect(lambda **_: setup(), weak=False)
    signals.worker_process_shutdown.connect(shutdown, weak=False)
    signals.worker_shutdown.connect(shutdown, weak=False)
    signals.after_setup_logger.connect(use_json_logs, weak=False)
    signals.after_setup_task_logger.connect(use_json_logs, weak=False)
    signals.task_prerun.connect(_task_prerun, weak=False)
    signals.task_postrun.connect(_task_postrun, weak=False)
