"""Table regions of an extraction run (IP-018).

The analyzer's layout pass finds the tables; here each is read with
pdfplumber inside its box, turned into markdown, cropped to WebP and kept
with its position, instead of being dropped as the indexing pipeline does.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import pdfplumber
import pymupdf

from evilflowers_text.extraction import geometry
from evilflowers_text.observability import metrics
from evilflowers_text.schemas import Box, LayoutRegion, Region

logger = logging.getLogger(__name__)

#: More rotated characters than this and the table is not read as text (landscape tables).
MAX_ROTATED_SHARE = 0.3
#: Below this share of filled cells the grid is mostly guesswork.
MIN_FILLED_SHARE = 0.35
#: Above this share of tiny cells ("Eve | nt", "ro") words were cut across columns.
MAX_FRAGMENT_SHARE = 0.4
#: pdfplumber's line-based default first, then the text-alignment strategy.
STRATEGIES: tuple[dict, ...] = ({}, {"vertical_strategy": "text", "horizontal_strategy": "text"})

Rows = list[list[str | None]]


def markdown(rows: Rows) -> str | None:
    """A GitHub-style markdown table; the first row is the header."""
    cells = [[" ".join((cell or "").split()).replace("|", "\\|") for cell in row] for row in rows if row]
    cells = [row for row in cells if any(row)]
    if not cells:
        return None
    width = max(len(row) for row in cells)
    cells = [row + [""] * (width - len(row)) for row in cells]
    lines = ["| " + " | ".join(cells[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in cells[1:]]
    return "\n".join(lines)


def rotated_share(chars: list[dict]) -> float:
    if not chars:
        return 0.0
    return sum(1 for char in chars if not char.get("upright", True)) / len(chars)


def plausible(rows: Rows) -> bool:
    """False when the grid cut words apart or is mostly empty: better no markdown than garbage."""
    cells = [" ".join((cell or "").split()) for row in rows for cell in row]
    filled = [cell for cell in cells if cell]
    if not cells or len(filled) / len(cells) < MIN_FILLED_SHARE:
        return False
    fragments = [cell for cell in filled if len(cell) <= 3 and cell.isalpha() and cell.islower()]
    return len(fragments) / len(filled) <= MAX_FRAGMENT_SHARE


def read_table(plumber_page, box: Box) -> str | None:
    """Markdown of the table inside `box`, or None when it cannot be read reliably.

    The crop is kept either way, so a reader still sees the table as an image.
    """
    width, height = float(plumber_page.width), float(plumber_page.height)
    bounds = (box.x0 * width, box.y0 * height, box.x1 * width, box.y1 * height)
    area = plumber_page.within_bbox(bounds, relative=False, strict=False)
    if rotated_share(area.chars) > MAX_ROTATED_SHARE:
        return None
    for strategy in STRATEGIES:
        table = area.extract_table(strategy) if strategy else area.extract_table()
        if table and plausible(table):
            return markdown(table)
    return None


class TableReader:
    """Reads the analyzer's table regions of one PDF into catalog regions with a WebP crop each."""

    def __init__(self, storage_path: Path, *, dpi: int, quality: int):
        self.storage_path = storage_path
        self.dpi = dpi
        self.quality = quality

    @staticmethod
    def versions() -> dict[str, str]:
        """The libraries behind the markdown and the crops, for the run's provenance."""
        return {"extractor": "pdfplumber", "pdfplumber": pdfplumber.__version__, "pymupdf": pymupdf.VersionBind}

    def read(self, path: Path, output_dir: str, regions: Sequence[LayoutRegion]) -> list[Region]:
        """One region per readable table; a region that fails is logged and skipped, never the document."""
        kept = []
        with pdfplumber.open(path) as plumber, pymupdf.open(path) as document:
            for region in regions:
                try:
                    kept.append(self._region(plumber, document, output_dir, region))
                except Exception:
                    logger.exception("Table on page %s skipped", region.page)
        return kept

    def _region(self, plumber, document: pymupdf.Document, output_dir: str, region: LayoutRegion) -> Region:
        page = document[region.page - 1]
        image = geometry.render(page, geometry.to_rect(region.bbox, page), self.dpi)
        name = geometry.crop_name(output_dir, region.page, "tab", region.seq)
        geometry.save(image, self.storage_path / name, self.quality)
        text = read_table(plumber.pages[region.page - 1], region.bbox)
        metrics.TABLES.add(1, {"markdown": "read" if text else "unreadable"})
        return Region(
            kind="table",
            page=region.page,
            seq=region.seq,
            bbox=region.bbox,
            image_path=name,
            caption=region.caption,
            printed_caption=region.caption,
            markdown=text,
        )
