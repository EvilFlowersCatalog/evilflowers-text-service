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


def read_table(plumber_page, box: dict) -> Optional[str]:
    """Markdown of the table inside `box` (page fractions), or None when no grid is found there."""
    width, height = float(plumber_page.width), float(plumber_page.height)
    bounds = (box["x0"] * width, box["y0"] * height, box["x1"] * width, box["y1"] * height)
    area = plumber_page.within_bbox(bounds, relative=False, strict=False)
    table = area.extract_table() or area.extract_table({"vertical_strategy": "text", "horizontal_strategy": "text"})
    return markdown(table) if table else None


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
