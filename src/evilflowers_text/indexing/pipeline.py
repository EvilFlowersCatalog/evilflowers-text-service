"""PDF → search-service chunks (`process_pdf`): text extraction, then chunking.

The heavy dependencies (OCR, layout model, llama-index) load on first use, so
a worker that only serves the IP-018 jobs never imports them.
"""

import logging
from pathlib import Path
from typing import Any

from evilflowers_text.indexing.chunker import Chunker
from evilflowers_text.indexing.tables import table_extractor
from evilflowers_text.indexing.text_extractor import TextExtractor
from evilflowers_text.observability import span
from evilflowers_text.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class IndexingPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        extractor: TextExtractor | None = None,
        chunker: Chunker | None = None,
    ):
        self.settings = settings or get_settings()
        self.extractor = extractor or TextExtractor(self.settings)
        self._chunker = chunker

    @property
    def chunker(self) -> Chunker:
        if self._chunker is None:
            self._chunker = Chunker(self.settings.chunk_size, self.settings.chunk_overlap)
        return self._chunker

    def chunks(self, path: Path, document_id: str) -> dict[str, Any]:
        """The `chunks` of the search-service `/index` payload for one PDF."""
        with span("text.extract") as current:
            pages, toc = self.extractor.extract(path)
            if current is not None:
                current.set_attribute("text.pages", len(pages))
        logger.debug("ToC of %s: %s", document_id, toc)
        with span("text.chunk") as current:
            result = self.chunker.chunk(pages, toc, doc_id=document_id)
            if current is not None:
                current.set_attribute("text.chunks", result["total_chunks"])
        return result

    def tables(self, path: Path) -> dict:
        """Every table of the PDF as a data frame (`TABLE_EXTRACTOR`)."""
        return table_extractor(self.settings.table_extractor, self.settings.table_extractor_mode).extract(path)
