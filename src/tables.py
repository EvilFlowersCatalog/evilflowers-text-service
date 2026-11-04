"""Table regions of an extraction run (IP-018).

The analyzer's layout pass finds the tables; here each is read with
pdfplumber inside its box, turned into markdown, cropped to WebP and kept
with its position, instead of being dropped as the indexing pipeline does.
"""

import logging
from typing import List, Optional

import fitz
import pdfplumber

import crops
import settings

logger = logging.getLogger(__name__)


def markdown(rows: List[List[Optional[str]]]) -> Optional[str]:
    """A GitHub-style markdown table; the first row is the header."""
    rows = [[" ".join((cell or "").split()).replace("|", "\\|") for cell in row] for row in rows if row]
    rows = [row for row in rows if any(row)]
    if not rows:
        return None
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


#: More rotated characters than this and the table is not read as text (landscape tables).
MAX_ROTATED_SHARE = 0.3
#: Below this share of filled cells the grid is mostly guesswork.
MIN_FILLED_SHARE = 0.35
#: Above this share of tiny cells ("Eve | nt", "ro") words were cut across columns.
MAX_FRAGMENT_SHARE = 0.4


def rotated_share(chars: List[dict]) -> float:
    if not chars:
        return 0.0
    return sum(1 for char in chars if not char.get("upright", True)) / len(chars)


def plausible(rows: List[List[Optional[str]]]) -> bool:
    """False when the grid cut words apart or is mostly empty: better no markdown than garbage."""
    cells = [" ".join((cell or "").split()) for row in rows for cell in row]
    filled = [cell for cell in cells if cell]
    if not cells or len(filled) / len(cells) < MIN_FILLED_SHARE:
        return False
    fragments = [cell for cell in filled if len(cell) <= 3 and cell.isalpha() and cell.islower()]
    return len(fragments) / len(filled) <= MAX_FRAGMENT_SHARE


def read_table(plumber_page, box: dict) -> Optional[str]:
    """Markdown of the table inside `box` (page fractions), or None when it cannot be read reliably.

    The crop is kept either way, so a reader still sees the table as an image.
    """
    width, height = float(plumber_page.width), float(plumber_page.height)
    bounds = (box["x0"] * width, box["y0"] * height, box["x1"] * width, box["y1"] * height)
    area = plumber_page.within_bbox(bounds, relative=False, strict=False)
    if rotated_share(area.chars) > MAX_ROTATED_SHARE:
        return None
    for strategy in ({}, {"vertical_strategy": "text", "horizontal_strategy": "text"}):
        table = area.extract_table(strategy) if strategy else area.extract_table()
        if table and plausible(table):
            return markdown(table)
    return None


def extract(path: str, output_dir: str, regions: List[dict]) -> List[dict]:
    kept = []
    with pdfplumber.open(path) as plumber, fitz.open(path) as document:
        for region in regions:
            try:
                page = document[region["page"] - 1]
                image = crops.render(page, crops.rect(region["bbox"], page), settings.CROP_DPI)
                name = crops.crop_name(output_dir, region["page"], "tab", region["seq"])
                crops.save(image, name)
                kept.append(
                    {
                        "kind": "table",
                        "page": region["page"],
                        "seq": region["seq"],
                        "bbox": region["bbox"],
                        "image_path": name,
                        "caption": region.get("caption"),
                        "markdown": read_table(plumber.pages[region["page"] - 1], region["bbox"]),
                    }
                )
            except Exception:
                logger.exception("Table on page %s skipped", region.get("page"))
    return kept
