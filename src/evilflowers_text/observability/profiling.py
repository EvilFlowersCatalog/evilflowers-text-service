"""Continuous CPU profiling through the Pyroscope SDK (Python frames).

The node's eBPF profiler cannot unwind Python on k3s ``control`` (knowledge
Known-Issues 6i), so the worker profiles itself. The contract is the catalog's
(EvilFlowersCatalog#111); the pod gets the variables from the knowledge repo
(``evilflowers_instances.<env>.pyroscope_sdk``), which also labels the pod so the
eBPF profiler skips it:

- ``PYROSCOPE_SERVER_ADDRESS``: profiling is on only when it is set;
- ``PYROSCOPE_APPLICATION_NAME``: the ``service_name`` in Pyroscope (``text``);
- ``PYROSCOPE_TAGS``: ``k=v`` pairs separated by commas
  (``namespace=…,pod=…,container=text,cluster=k8s-stu``).

The profiler samples the process that calls ``setup()``. It runs from
``worker_process_init`` (see ``signals``): the worker itself under ``--pool=solo``,
each child after the fork under prefork.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

DEFAULT_APPLICATION_NAME = "text"
SAMPLE_RATE = 100

_lock = threading.Lock()
_state = {"done": False, "enabled": False}


def parse_tags(value: str | None) -> dict[str, str]:
    """``"a=1, b=2"`` → ``{"a": "1", "b": "2"}``; malformed pairs are skipped with a warning."""
    tags: dict[str, str] = {}
    for pair in (value or "").split(","):
        if not pair.strip():
            continue
        key, separator, tag_value = pair.partition("=")
        key, tag_value = key.strip(), tag_value.strip()
        if not separator or not key or not tag_value:
            logger.warning("Ignoring malformed PYROSCOPE_TAGS entry %r", pair.strip())
            continue
        tags[key] = tag_value
    return tags


def is_enabled() -> bool:
    return bool(_state["enabled"])


def setup() -> bool:
    """Start the profiler when ``PYROSCOPE_SERVER_ADDRESS`` is set. Idempotent; returns whether it runs."""
    with _lock:
        if _state["done"]:
            return _state["enabled"]
        _state["done"] = True

        server_address = os.environ.get("PYROSCOPE_SERVER_ADDRESS", "").strip()
        if not server_address:
            logger.info("Pyroscope profiling off: no PYROSCOPE_SERVER_ADDRESS")
            return False
        try:
            import pyroscope
        except ImportError as error:
            logger.warning("pyroscope-io is not installed (%s); profiling is off", error)
            return False

        application_name = os.environ.get("PYROSCOPE_APPLICATION_NAME", "").strip() or DEFAULT_APPLICATION_NAME
        tags = parse_tags(os.environ.get("PYROSCOPE_TAGS"))
        try:
            pyroscope.configure(
                application_name=application_name,
                server_address=server_address,
                oncpu=True,
                gil_only=False,
                sample_rate=SAMPLE_RATE,
                tags=tags,
            )
        except Exception:
            # Profiling is diagnostics; it must never keep the worker from taking tasks.
            logger.exception("Pyroscope profiling failed to start")
            return False
        _state["enabled"] = True
        logger.info("Pyroscope profiling on: server=%s application=%s tags=%s", server_address, application_name, tags)
        return True


def span_processor():
    """``PyroscopeSpanProcessor`` (links Tempo spans to their profiles), or ``None`` when profiling is off."""
    if not is_enabled():
        return None
    try:
        from pyroscope.otel import PyroscopeSpanProcessor
    except ImportError:
        logger.info("pyroscope-otel is not installed; spans are not linked to profiles")
        return None
    return PyroscopeSpanProcessor()


def shutdown() -> None:
    """Flush the last profile and stop the profiler. Idempotent; never fails the exit path."""
    with _lock:
        enabled, _state["enabled"] = _state["enabled"], False
    if not enabled:
        return
    try:
        import pyroscope

        pyroscope.shutdown()
    except Exception:  # pragma: no cover
        logger.debug("Pyroscope shutdown failed", exc_info=True)
