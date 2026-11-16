"""The two IP-018 leaf jobs of text: read one PDF and post the results to the catalog.

The lifecycle is the kit's ``evilflowers_worker.jobs.LeafJob.run`` (issue #16): the delivery cap, the read of the
run before the work (a copy, a run no longer running or a superseded attempt stops there), ``extract``, ``publish``
with ``?attempt=``, and the terminal report, under the message's ``job`` when it names a batch
(``tables:0001-0008``) and under the kind otherwise. Here a job only says how it extracts, what it publishes and what
provenance it records.
"""

import logging
from typing import Any

import pymupdf

from evilflowers_text import embeddings, metrics
from evilflowers_text.embeddings import Encoder
from evilflowers_text.extraction import paragraphs
from evilflowers_text.extraction.decider import TableDecider
from evilflowers_text.extraction.table_review import TableReviewer
from evilflowers_text.extraction.tables import Decision, TableReader
from evilflowers_text.settings import Settings
from evilflowers_worker import contract
from evilflowers_worker.contract import Paragraph, ParagraphLink, Region, RegionsJob
from evilflowers_worker.jobs import LeafJob
from evilflowers_worker.judges import Judges
from evilflowers_worker.observability import span

logger = logging.getLogger(__name__)

ParagraphIndex = tuple[list[Paragraph], list[ParagraphLink]]

_judges: tuple[Settings, Judges] | None = None


def process_judges(settings: Settings) -> Judges:
    """The process's judge clients: their circuit breakers outlive one job, so they are built once per settings."""
    global _judges
    if _judges is None or _judges[0] is not settings:
        _judges = (settings, Judges.from_settings(settings))
    return _judges[1]


class TablesJob(LeafJob[RegionsJob, list[Region]]):
    """Table regions found by the analyzer's layout pass: markdown, printed caption, crop and position."""

    kind = "tables"
    settings: Settings

    def __init__(self, message: RegionsJob, *, settings: Settings, judges: Judges | None = None, **kwargs: Any):
        super().__init__(message, settings=settings, **kwargs)
        reviewer = None
        if settings.table_review_enabled and settings.ollama_endpoint:
            key = settings.ollama_api_key
            reviewer = TableReviewer(
                settings.ollama_endpoint,
                settings.table_review_model,
                api_key=key.get_secret_value() if key else None,
                timeout=settings.table_review_timeout,
                max_side=settings.table_review_max_side,
            )
        judges = judges or process_judges(settings)
        decider = None
        if judges.decider_vision is not None:
            # One panel per job: it holds this job's budget of judge calls.
            decider = TableDecider(judges.decider_vision, judges.panel())
        self.reader = TableReader(
            settings.storage_path,
            dpi=settings.crop_dpi,
            quality=settings.webp_quality,
            reviewer=reviewer,
            decider=decider,
            decision=Decision(settings.decider_table_bands),
        )

    def extract(self) -> list[Region]:
        message = self.message
        # The crops are written under `output_dir`: a path from the message must not leave the storage root.
        self.settings.storage_file(message.output_dir)
        with span("tables.extract", **{"tables.regions": len(message.regions)}) as current:
            kept = self.reader.read(self.settings.source_path(message.source), message.output_dir, message.regions)
            if current is not None:
                current.set_attribute("tables.kept", len(kept))
        return kept

    def publish(self, result: list[Region]) -> None:
        self.catalog.regions(result)

    def provenance(self) -> dict[str, Any]:
        return {**super().provenance(), **self.reader.versions(), "crop_dpi": self.settings.crop_dpi}


class ParagraphsJob(LeafJob[contract.ParagraphsJob, ParagraphIndex]):
    """The paragraph index (IP-018 Phase 2b): paragraphs, sentences, embeddings and links."""

    kind = "paragraphs"
    settings: Settings

    def __init__(
        self, message: contract.ParagraphsJob, *, settings: Settings, encoder: Encoder | None = None, **kwargs: Any
    ):
        kwargs.setdefault("recheck_interval", settings.run_recheck_interval)
        super().__init__(message, settings=settings, **kwargs)
        self._encoder = encoder

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = embeddings.encoder()
        return self._encoder

    @property
    def outline_source(self) -> str:
        """Where the sections came from: the analyzer's outline (issue #15) or the PDF's bookmarks."""
        return "bookmarks" if self.message.outline is None else "analyzer"

    def extract(self) -> ParagraphIndex:
        message = self.message
        path = self.settings.source_path(message.source)
        with span("paragraphs.extract") as current, pymupdf.open(path) as document:
            rows = paragraphs.extract(document, self.settings.min_paragraph_words, message.outline)
            if current is not None:
                current.set_attribute("paragraphs.count", len(rows))
                current.set_attribute("paragraphs.outline", self.outline_source)
        if not rows:
            return [], []
        # Embedding a book takes minutes to hours: between batches, stop if the run moved on (#14, catalog#140).
        vectors = self.encoder.documents([paragraphs.embedding_text(row) for row in rows], between=self.recheck)
        for row, vector in zip(rows, vectors, strict=True):
            row.embedding = vector
        with span("paragraphs.links", **{"paragraphs.count": len(vectors)}):
            links = paragraphs.links(vectors, self.settings.similar_links_k, self.settings.similar_min_score)
        return rows, links

    def publish(self, result: ParagraphIndex) -> None:
        rows, links = result
        self.catalog.paragraphs(rows, links)
        metrics.PARAGRAPHS.record(len(rows))
        logger.info("Run %s: %d paragraphs, %d links", self.message.run_id, len(rows), len(links))

    def provenance(self) -> dict[str, Any]:
        base = {**super().provenance(), "outline": self.outline_source}
        # `model_id@model_sha`: the catalog never compares vectors of two models.
        try:
            model = self.encoder.name
        except Exception:
            # Also called to report a failure, often because tei-embed is down: that report must still go out.
            logger.warning("Embedding model unknown for run %s", self.message.run_id, exc_info=True)
            return base
        return {**base, "embedding_model": model}
