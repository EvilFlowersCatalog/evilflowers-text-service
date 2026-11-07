import json
import logging

import pytest

import telemetry

ENDPOINT_VARIABLES = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "HOST_IP",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENDPOINT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def record(message="hello", exc_info=None):
    return logging.LogRecord("text", logging.WARNING, __file__, 1, message, None, exc_info)


def test_no_endpoint_without_configuration(clean_env):
    assert telemetry.endpoint() is None


def test_explicit_endpoint_wins(clean_env):
    clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    clean_env.setenv("HOST_IP", "10.0.0.1")
    assert telemetry.endpoint() == "http://collector:4318"


def test_endpoint_from_host_ip(clean_env):
    clean_env.setenv("HOST_IP", "192.168.122.10")
    assert telemetry.endpoint() == "http://192.168.122.10:4318"


def test_endpoint_from_ipv6_host_ip(clean_env):
    clean_env.setenv("HOST_IP", "fd00::1")
    assert telemetry.endpoint() == "http://[fd00::1]:4318"


def test_span_and_metrics_are_safe_before_setup():
    with telemetry.span("test.span", **{"absent": None, "present": 1}):
        telemetry.annotate(**{"key": "value"})
        telemetry.fail(RuntimeError("boom"))
    telemetry.record_embedding("query", "model", 1, 0.01)
    telemetry.record_job("paragraphs", "done")


def test_json_log_line_outside_a_span():
    line = json.loads(telemetry.JsonFormatter().format(record("hello %s" % "world")))
    assert line["level"] == "warning"
    assert line["logger"] == "text"
    assert line["message"] == "hello world"
    assert "trace_id" not in line


def test_json_log_line_carries_the_traceback():
    try:
        raise ValueError("bad pdf")
    except ValueError:
        import sys

        line = json.loads(telemetry.JsonFormatter().format(record("failed", sys.exc_info())))
    assert "ValueError: bad pdf" in line["exception"]


def test_json_log_line_inside_a_span_has_the_trace_id():
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    tracer = sdk.TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("work") as span:
        line = json.loads(telemetry.JsonFormatter().format(record()))
        expected = format(span.get_span_context().trace_id, "032x")
    assert line["trace_id"] == expected
    assert len(line["span_id"]) == 16
