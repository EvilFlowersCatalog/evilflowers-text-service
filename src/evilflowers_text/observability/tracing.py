"""Hand-made spans where the time goes (PDF parsing, model load, encoding); no-ops without OpenTelemetry."""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

from evilflowers_text.observability.context import clean
from evilflowers_text.observability.otel import SCOPE, Status, StatusCode, trace

try:  # Ships with the `telemetry` extra (the requests instrumentation), not with the bare API.
    from opentelemetry.instrumentation.utils import suppress_http_instrumentation
except ImportError:  # pragma: no cover - the image installs the extra
    suppress_http_instrumentation = None


def untraced_http() -> AbstractContextManager[Any]:
    """No CLIENT span (and no ``traceparent``) for the ``requests`` calls in the block.

    For request loops whose per-call spans would swamp the trace (one task, hundreds
    of TEI batches): the block's own span carries the totals, the metrics the per-call
    timings, and the server side traces (and samples) its requests as roots of its own.
    """
    return suppress_http_instrumentation() if suppress_http_instrumentation is not None else nullcontext()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """A child span of the current one, or ``None``. An exception escaping the block marks it failed."""
    if trace is None:
        yield None
        return
    with trace.get_tracer(SCOPE).start_as_current_span(name, attributes=clean(attributes)) as current:
        yield current


def annotate(**attributes: Any) -> None:
    """Attributes on the current span."""
    if trace is None:
        return
    current = trace.get_current_span()
    for key, value in clean(attributes).items():
        current.set_attribute(key, value)


def fail(error: BaseException) -> None:
    """Mark the current span failed for an exception the code catches (and reports) itself.

    The extraction jobs report their own failures to the catalog and return, so
    without this the Celery span would end green.
    """
    if trace is None:
        return
    current = trace.get_current_span()
    current.record_exception(error)
    current.set_status(Status(StatusCode.ERROR, str(error)[:200]))
