"""OpenTelemetry providers of the worker process: endpoint discovery, resource, setup and shutdown.

The contract is the cluster's (knowledge base, Cookbooks/Deploy-a-Service §3):

- configuration is the standard ``OTEL_*`` environment. The pod sets ``HOST_IP``
  (downward API) and ``OTEL_EXPORTER_OTLP_ENDPOINT=http://$(HOST_IP):4318``;
  when only ``HOST_IP`` is set, the endpoint is derived from it. With neither,
  telemetry stays off and the worker behaves as if this package did not exist;
- traces and metrics go over OTLP/HTTP to the node's Alloy agent;
- logs are NOT exported over OTLP: Alloy tails stdout (see ``logs``).

What is instrumented once the providers are up:

- every task (CONSUMER span; continues the producer's trace when the caller,
  catalog or analyzer, propagates the context in the task headers);
- every ``requests`` call (catalog ingestion API, search-service): CLIENT spans
  carrying ``traceparent``, so the catalog's Django span joins the same trace;
- process metrics (CPU, resident memory: what a loaded embedding model costs).

Redis is deliberately not instrumented: the Celery broker polls it constantly
and every poll would become a root trace.
"""

import logging
import os
import socket
import threading

from evilflowers_text.observability.otel import OTEL_AVAILABLE, metrics, trace
from evilflowers_text.settings import get_settings

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "evilflowers-text"
OTLP_HTTP_PORT = 4318

ENVIRONMENT_DEFAULTS = {
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    "OTEL_SEMCONV_STABILITY_OPT_IN": "http",
    # The k8s attributes (pod, node, namespace) are added by Alloy's k8sattributes processor.
    "OTEL_EXPERIMENTAL_RESOURCE_DETECTORS": "process,os,host",
}

# Process metrics only: container CPU/memory come from cAdvisor, profiles from Pyroscope.
SYSTEM_METRICS_CONFIG = {
    "process.cpu.time": ["user", "system"],
    "process.memory.usage": None,
    "process.thread.count": None,
    "cpython.gc.collections": None,
}

ENDPOINT_VARIABLES = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
)

_lock = threading.Lock()
_state = {"done": False, "enabled": False}
_providers: list = []


def endpoint() -> str | None:
    """The OTLP endpoint: explicit ``OTEL_EXPORTER_OTLP_*ENDPOINT``, else derived from ``HOST_IP``."""
    for name in ENDPOINT_VARIABLES:
        if value := os.environ.get(name, "").strip():
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

        if not OTEL_AVAILABLE or os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
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

    settings = get_settings()
    from_environment = OTELResourceDetector().detect().attributes
    defaults = {
        "service.name": os.environ.get("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME,
        "service.namespace": "elvira",
        "service.version": settings.image_tag,
        "service.instance.id": f"{socket.gethostname()}-{os.getpid()}",
        "evilflowers.text.embedding_model": settings.embedding_model,
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
