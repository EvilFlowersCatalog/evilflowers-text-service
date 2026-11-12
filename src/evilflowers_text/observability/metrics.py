"""The worker's metrics, declared once; each is created on first use and is a no-op without OpenTelemetry.

Attributes stay low-cardinality: never a run id or document id (those go on spans and log lines).
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from evilflowers_text.observability.otel import SCOPE, metrics

# Seconds. Covers a 20 ms query embedding up to a 30 min book.
DURATION_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800)
COUNT_BUCKETS = (0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

_lock = threading.Lock()


@dataclass(eq=False)
class Instrument:
    kind: Literal["counter", "histogram"]
    name: str
    unit: str
    description: str
    buckets: tuple[float, ...] | None = None
    _created: Any = field(default=None, init=False, repr=False)

    def _get(self):
        if metrics is None:
            return None
        if self._created is None:
            with _lock:
                if self._created is None:
                    # A proxy meter: its instruments become real once `setup()` sets the provider.
                    meter = metrics.get_meter(SCOPE)
                    if self.kind == "counter":
                        self._created = meter.create_counter(self.name, self.unit, self.description)
                    else:
                        self._created = meter.create_histogram(
                            self.name,
                            self.unit,
                            self.description,
                            explicit_bucket_boundaries_advisory=list(self.buckets or DURATION_BUCKETS),
                        )
        return self._created

    def add(self, value: float = 1, attributes: dict[str, Any] | None = None) -> None:
        if (instrument := self._get()) is not None:
            instrument.add(value, attributes or {})

    def record(self, value: float, attributes: dict[str, Any] | None = None) -> None:
        if (instrument := self._get()) is not None:
            instrument.record(value, attributes or {})


TASK_DURATION = Instrument("histogram", "text.task.duration", "s", "Celery task run time")
JOB_OUTCOME = Instrument("counter", "text.job.outcome", "{job}", "IP-018 jobs reported to the catalog, by outcome")
EMBEDDING_DURATION = Instrument(
    "histogram", "text.embedding.duration", "s", "Time to embed one batch of paragraphs through tei-embed"
)
EMBEDDING_INPUTS = Instrument("counter", "text.embedding.inputs", "{text}", "Texts embedded")
PARAGRAPHS = Instrument(
    "histogram", "text.document.paragraphs", "{paragraph}", "Paragraphs indexed per document", COUNT_BUCKETS
)
TABLES = Instrument(
    "counter", "text.tables.regions", "{table}", "Table regions read, by whether markdown could be read"
)
CATALOG_RETRIES = Instrument(
    "counter", "text.catalog.retries", "{retry}", "Retried calls to the catalog ingestion API"
)


def record_job(job: str, outcome: str) -> None:
    """``outcome``: done, failed or not_running (the catalog refused the results)."""
    JOB_OUTCOME.add(1, {"job": job, "outcome": outcome})


def record_embedding(kind: str, model: str, inputs: int, seconds: float, lane: str | None = None) -> None:
    """``kind``: documents (paragraph batches). ``lane``: the embed lane that served the batch, if the router said."""
    attributes = {"kind": kind, "model": model}
    if lane:
        attributes["lane"] = lane
    EMBEDDING_DURATION.record(seconds, attributes)
    EMBEDDING_INPUTS.add(inputs, attributes)
