"""Page geometry and crops shared by the region jobs (IP-018).

Boxes arrive as page fractions with a top-left origin on the original file's
1-based page, the same convention PyMuPDF uses for page coordinates. Crops are
written as WebP into the run's folder under the shared storage mount.
"""

import io
from pathlib import Path

import pymupdf
from PIL import Image

from evilflowers_text.schemas import Box


def _clamp(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 5)


def to_box(rect: pymupdf.Rect, width: float, height: float) -> Box:
    """A rectangle in page points as page fractions."""
    return Box(
        x0=_clamp(rect.x0 / width),
        y0=_clamp(rect.y0 / height),
        x1=_clamp(rect.x1 / width),
        y1=_clamp(rect.y1 / height),
    )


def to_rect(box: Box, page: pymupdf.Page) -> pymupdf.Rect:
    """The box in the page's points."""
    width, height = page.rect.width, page.rect.height
    return pymupdf.Rect(box.x0 * width, box.y0 * height, box.x1 * width, box.y1 * height)


def render(page: pymupdf.Page, clip: pymupdf.Rect, dpi: int) -> Image.Image:
    pixmap = page.get_pixmap(clip=clip, dpi=dpi, alpha=False)
    return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")


def crop_name(output_dir: str, page: int, kind: str, seq: int) -> str:
    """Storage name of a crop: ``<output_dir>/p012-tab03.webp``."""
    return f"{output_dir.rstrip('/')}/p{page:03d}-{kind}{seq:02d}.webp"


def save(image: Image.Image, path: Path, quality: int) -> tuple[int, int]:
    """Write the crop; readable by the catalog, which serves and deletes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "WEBP", quality=quality, method=4)
    return image.size
