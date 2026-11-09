import sys
import types

import pytest

from evilflowers_text.observability import profiling


@pytest.fixture
def fresh(monkeypatch):
    """Profiling state reset, no PYROSCOPE_* variables, and a fake ``pyroscope`` that records its calls."""
    monkeypatch.setattr(profiling, "_state", {"done": False, "enabled": False})
    for name in ("PYROSCOPE_SERVER_ADDRESS", "PYROSCOPE_APPLICATION_NAME", "PYROSCOPE_TAGS"):
        monkeypatch.delenv(name, raising=False)
    fake = types.ModuleType("pyroscope")
    fake.calls = []
    fake.configure = lambda **kwargs: fake.calls.append(("configure", kwargs))
    fake.shutdown = lambda: fake.calls.append(("shutdown", {}))
    monkeypatch.setitem(sys.modules, "pyroscope", fake)
    return monkeypatch, fake


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, {}),
        ("", {}),
        (
            "namespace=evilflowers-dev,pod=text-abc,container=text,cluster=k8s-stu",
            {"namespace": "evilflowers-dev", "pod": "text-abc", "container": "text", "cluster": "k8s-stu"},
        ),
        (" a = 1 , ,b=2,", {"a": "1", "b": "2"}),
        ("a=1,broken,=x,y=,c=d=e", {"a": "1", "c": "d=e"}),
    ],
)
def test_parse_tags(value, expected):
    assert profiling.parse_tags(value) == expected


def test_off_without_server_address(fresh):
    _, fake = fresh
    assert profiling.setup() is False
    assert not profiling.is_enabled()
    assert profiling.span_processor() is None
    profiling.shutdown()
    assert fake.calls == []


def test_configures_pyroscope_from_the_environment(fresh):
    monkeypatch, fake = fresh
    monkeypatch.setenv("PYROSCOPE_SERVER_ADDRESS", "http://pyroscope:4040")
    monkeypatch.setenv("PYROSCOPE_APPLICATION_NAME", "text")
    monkeypatch.setenv("PYROSCOPE_TAGS", "namespace=evilflowers-dev,container=text")

    assert profiling.setup() is True
    assert profiling.setup() is True  # idempotent
    assert fake.calls == [
        (
            "configure",
            {
                "application_name": "text",
                "server_address": "http://pyroscope:4040",
                "oncpu": True,
                "gil_only": False,
                "sample_rate": 100,
                "tags": {"namespace": "evilflowers-dev", "container": "text"},
            },
        )
    ]

    profiling.shutdown()
    profiling.shutdown()  # idempotent
    assert [call for call, _ in fake.calls].count("shutdown") == 1
    assert not profiling.is_enabled()


def test_application_name_defaults_to_text(fresh):
    monkeypatch, fake = fresh
    monkeypatch.setenv("PYROSCOPE_SERVER_ADDRESS", "http://pyroscope:4040")
    profiling.setup()
    assert fake.calls[0][1]["application_name"] == "text"


def test_a_failing_configure_leaves_profiling_off(fresh):
    monkeypatch, fake = fresh
    monkeypatch.setenv("PYROSCOPE_SERVER_ADDRESS", "http://pyroscope:4040")

    def explode(**_kwargs):
        raise RuntimeError("no agent")

    fake.configure = explode
    assert profiling.setup() is False
    assert not profiling.is_enabled()


def test_span_processor_when_profiling_is_on(monkeypatch):
    otel = pytest.importorskip("pyroscope.otel", exc_type=ImportError)
    monkeypatch.setattr(profiling, "_state", {"done": True, "enabled": True})
    assert isinstance(profiling.span_processor(), otel.PyroscopeSpanProcessor)
