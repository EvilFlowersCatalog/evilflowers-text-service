"""JSON log lines: Alloy tails stdout, so every line is one object Loki can parse.

Fields: ``time``, ``level``, ``logger``, ``message``; inside a span ``trace_id``
and ``span_id`` (the Loki → Tempo link); inside a task ``task_name``,
``task_id`` and what the task bound (``run_id``, ``entry_id``, ``source`` …);
and the traceback.
"""

import json
import logging
from datetime import UTC, datetime

from evilflowers_text.observability.context import current
from evilflowers_text.observability.otel import trace
from evilflowers_text.settings import get_settings

#: Keys a bound field may not overwrite.
_RESERVED = frozenset({"time", "level", "logger", "message", "trace_id", "span_id", "task_name", "task_id"})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if trace is not None:
            context = trace.get_current_span().get_span_context()
            if context.is_valid:
                payload["trace_id"] = format(context.trace_id, "032x")
                payload["span_id"] = format(context.span_id, "016x")
        if (task := _current_task()) is not None:
            payload["task_name"], payload["task_id"] = task
        payload.update({key: value for key, value in current().items() if key not in _RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _current_task() -> tuple[str, str] | None:
    try:
        from celery import current_task

        if current_task and current_task.request.id:
            return current_task.name, current_task.request.id
    except Exception:  # pragma: no cover - outside a worker
        pass
    return None


def use_json_logs(logger: logging.Logger | None = None, **_kwargs) -> None:
    """Celery's ``after_setup_logger`` / ``after_setup_task_logger`` receiver."""
    if get_settings().log_format != "json":
        return
    for handler in (logger or logging.getLogger()).handlers:
        handler.setFormatter(JsonFormatter())
