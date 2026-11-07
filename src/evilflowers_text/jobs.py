"""IP-018 extraction jobs: read one PDF, post the results to the catalog, report the job.

The analyzer dispatches the jobs and declares them at the run's `/start`; the
run finishes only when every declared job has reported `done` or `failed`. So
a job never ends silently: whatever goes wrong is reported to the catalog,
unless the catalog itself refused the run (409, `RunNotRunning`).

`ExtractionJob.run` is that protocol, once. A job only says how it extracts,
what it publishes and what provenance it records.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import pymupdf
from pydantic import TypeAdapter

from evilflowers_text import embeddings
from evilflowers_text.clients.catalog import CatalogClient, RunNotRunning
from evilflowers_text.embeddings import Encoder
from evilflowers_text.extraction import paragraphs
from evilflowers_text.extraction.tables import TableReader
from evilflowers_text.observability import bind, fail, metrics, span
from evilflowers_text.schemas import JobReport, LayoutRegion, Paragraph, ParagraphLink, Region
from evilflowers_text.settings import Settings, get_settings

logger = logging.getLogger(__name__)

Outcome = Literal["done", "failed", "not_running"]


class ExtractionJob[Result](ABC):
    #: The job's name in the catalog (`/jobs/<name>`, `run.provenance[<name>]`).
    name: ClassVar[str]

    def __init__(
        self, run_id: str, source: str, *, settings: Settings | None = None, catalog: CatalogClient | None = None
    ):
        self.settings = settings or get_settings()
        self.run_id = run_id
        self.source = source
        self.path = self.settings.source_path(source)
        self.catalog = catalog or CatalogClient(run_id, settings=self.settings)

    @abstractmethod
    def extract(self) -> Result:
        """Read the PDF. Raising fails the job."""

    @abstractmethod
    def publish(self, result: Result) -> None:
        """Send the results to the catalog."""

    def provenance(self) -> dict[str, Any]:
        """What produced the results, stored as `run.provenance[<name>]`."""
        return {"image": self.settings.image_tag}

    def done(self, result: Result) -> None:  # noqa: B027 - an optional hook, not part of the contract
        """Hook after the catalog accepted the job."""

    def run(self) -> Outcome:
        with bind(run_id=self.run_id, job=self.name, source=self.source):
            try:
                result = self.extract()
                self.publish(result)
                self.catalog.job(self.name, JobReport(status="done", provenance=self.provenance()))
            except RunNotRunning:
                logger.warning("Run %s is not accepting results", self.run_id)
                return self._outcome("not_running")
            except Exception as error:
                logger.exception("Job %s failed for run %s", self.name, self.run_id)
                fail(error)
                self._report_failure(error)
                return self._outcome("failed")
            self.done(result)
            return self._outcome("done")

    def _report_failure(self, error: Exception) -> None:
        # Report it, or the run waits for this job forever.
        report = JobReport(status="failed", error=f"{type(error).__name__}: {error}", provenance=self.provenance())
        try:
            self.catalog.job(self.name, report)
        except RunNotRunning:
            logger.warning("Run %s is not accepting results", self.run_id)
        except Exception:
            logger.exception("Could not report the failed %s job for run %s", self.name, self.run_id)

    def _outcome(self, outcome: Outcome) -> Outcome:
        metrics.record_job(self.name, outcome)
        return outcome


class TablesJob(ExtractionJob[list[Region]]):
    """Table regions found by the analyzer's layout pass: markdown, printed caption, crop and position."""

    name = "tables"
    _regions = TypeAdapter(list[LayoutRegion])

    def __init__(self, run_id: str, source: str, output_dir: str, regions: Sequence[dict], **kwargs: Any):
        super().__init__(run_id, source, **kwargs)
        self.output_dir = output_dir
        self.raw_regions = regions
        self.reader = TableReader(
            self.settings.storage_path, dpi=self.settings.crop_dpi, quality=self.settings.webp_quality
        )

    def extract(self) -> list[Region]:
        regions = self._regions.validate_python(self.raw_regions)
        with span("tables.extract", **{"tables.regions": len(regions)}) as current:
            kept = self.reader.read(self.path, self.output_dir, regions)
            if current is not None:
                current.set_attribute("tables.kept", len(kept))
        return kept

    def publish(self, result: list[Region]) -> None:
        self.catalog.regions(result)

    def provenance(self) -> dict[str, Any]:
        return {**super().provenance(), **self.reader.versions(), "crop_dpi": self.settings.crop_dpi}


class ParagraphsJob(ExtractionJob[tuple[list[Paragraph], list[ParagraphLink]]]):
    """The paragraph index (IP-018 Phase 2b): paragraphs, sentences, embeddings and links."""

    name = "paragraphs"

    def __init__(self, run_id: str, source: str, *, encoder: Encoder | None = None, **kwargs: Any):
        super().__init__(run_id, source, **kwargs)
        self._encoder = encoder

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = embeddings.encoder(self.settings.embedding_model)
        return self._encoder

    def extract(self) -> tuple[list[Paragraph], list[ParagraphLink]]:
        with span("paragraphs.extract") as current, pymupdf.open(self.path) as document:
            rows = paragraphs.extract(document, self.settings.min_paragraph_words)
            if current is not None:
                current.set_attribute("paragraphs.count", len(rows))
        if not rows:
            return [], []
        vectors = self.encoder.documents([row.embedding_text for row in rows])
        for row, vector in zip(rows, vectors, strict=True):
            row.embedding = vector
        with span("paragraphs.links", **{"paragraphs.count": len(vectors)}):
            links = paragraphs.links(vectors, self.settings.similar_links_k, self.settings.similar_min_score)
        return rows, links

    def publish(self, result: tuple[list[Paragraph], list[ParagraphLink]]) -> None:
        self.catalog.paragraphs(*result)

    def provenance(self) -> dict[str, Any]:
        # The catalog embeds a reader's query with this model (`ReaderExtraction.embedding_model`).
        return {**super().provenance(), "embedding_model": self.settings.embedding_model}

    def done(self, result: tuple[list[Paragraph], list[ParagraphLink]]) -> None:
        rows, links = result
        metrics.PARAGRAPHS.record(len(rows))
        logger.info("Run %s: %d paragraphs, %d links", self.run_id, len(rows), len(links))
