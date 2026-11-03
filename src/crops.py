"""Page geometry and crops shared by the region services (IP-018).

Boxes arrive as page fractions with a top-left origin on the original file's
1-based page, the same convention PyMuPDF uses for page coordinates. Crops are
written as WebP into the run's folder under the shared storage mount.
"""

import io
import os
from typing import Dict, Tuple

import fitz
from PIL import Image

import settings


def rect(box: Dict[str, float], page: fitz.Page) -> fitz.Rect:
    """The box in the page's points."""
    width, height = page.rect.width, page.rect.height
    return fitz.Rect(box["x0"] * width, box["y0"] * height, box["x1"] * width, box["y1"] * height)


def coverage(box: Dict[str, float]) -> float:
    """Share of the page the box covers."""
    return max(box["x1"] - box["x0"], 0.0) * max(box["y1"] - box["y0"], 0.0)


def render(page: fitz.Page, clip: fitz.Rect, dpi: int) -> Image.Image:
    pixmap = page.get_pixmap(clip=clip, dpi=dpi, alpha=False)
    return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")


def crop_name(output_dir: str, page: int, kind: str, seq: int) -> str:
    """Storage name of a crop: `<output_dir>/p012-fig03.webp`."""
    return f"{output_dir.rstrip('/')}/p{page:03d}-{kind}{seq:02d}.webp"


def save(image: Image.Image, name: str) -> Tuple[int, int]:
    """Write the crop under the storage root; readable by the catalog, which serves and deletes it."""
    path = os.path.join(settings.STORAGE_PATH, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, "WEBP", quality=settings.WEBP_QUALITY, method=4)
    return image.size
