"""Text's own metrics, beside the kit's standard set (task duration, job outcome, catalog retries, capped deliveries).

``{service}`` is ``text``, so the names in Mimir are the ones dashboards already use
(``text_embedding_duration_seconds``).
Attributes stay low-cardinality: never a run id or document id (those go on spans and log lines).
"""

from evilflowers_worker.observability.metrics import COUNT_BUCKETS, Instrument

#: Seconds: a batch of 32 paragraphs is tens of milliseconds to a minute.
EMBEDDING_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800)

EMBEDDING_DURATION = Instrument(
    "histogram",
    "{service}.embedding.duration",
    "s",
    "Time to embed one batch of paragraphs through tei-embed",
    EMBEDDING_BUCKETS,
)
EMBEDDING_INPUTS = Instrument("counter", "{service}.embedding.inputs", "{text}", "Texts embedded")
PARAGRAPHS = Instrument(
    "histogram", "{service}.document.paragraphs", "{paragraph}", "Paragraphs indexed per document", COUNT_BUCKETS
)
TABLES = Instrument(
    "counter", "{service}.tables.regions", "{table}", "Table regions read, by whether markdown could be read"
)
#: ``outcome``: kept, rewritten, transcribed, unusable or failed (issue #19); ``text_table_review_total`` in Mimir.
TABLE_REVIEWS = Instrument("counter", "{service}.table.review", "{table}", "Table reviews by the model, by outcome")


def record_embedding(kind: str, model: str, inputs: int, seconds: float, lane: str | None = None) -> None:
    """``kind``: documents (paragraph batches). ``lane``: the embed lane that served the batch, if the router said."""
    attributes = {"kind": kind, "model": model}
    if lane:
        attributes["lane"] = lane
    EMBEDDING_DURATION.record(seconds, attributes)
    EMBEDDING_INPUTS.add(inputs, attributes)
