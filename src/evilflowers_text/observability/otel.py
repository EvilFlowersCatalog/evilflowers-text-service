"""The OpenTelemetry API, or ``None`` where it is not installed (the code degrades to no-ops)."""

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:  # pragma: no cover - the image always ships the API
    metrics = trace = Status = StatusCode = None

OTEL_AVAILABLE = trace is not None

#: Instrumentation scope of every span and instrument this worker creates.
SCOPE = "evilflowers.text"
