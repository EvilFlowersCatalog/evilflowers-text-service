"""Traces, metrics and JSON logs of the text worker (the cluster's telemetry contract, IP-019).

Off unless the pod points it at a collector; every call here is safe either way.
"""

from evilflowers_text.observability import metrics
from evilflowers_text.observability.context import bind
from evilflowers_text.observability.logs import JsonFormatter
from evilflowers_text.observability.setup import endpoint, is_enabled, setup, shutdown
from evilflowers_text.observability.tracing import annotate, fail, span

__all__ = [
    "JsonFormatter",
    "annotate",
    "bind",
    "endpoint",
    "fail",
    "is_enabled",
    "metrics",
    "setup",
    "shutdown",
    "span",
]
