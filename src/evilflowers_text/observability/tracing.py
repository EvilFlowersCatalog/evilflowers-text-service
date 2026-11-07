"""Hand-made spans where the time goes (PDF parsing, model load, encoding); no-ops without OpenTelemetry."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from evilflowers_text.observability.context import clean
from evilflowers_text.observability.otel import SCOPE, Status, StatusCode, trace


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
