"""Page text and table of contents of a PDF, with OCR for scans (search-service indexing)."""

import logging
import random
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from evilflowers_text.settings import Settings

logger = logging.getLogger(__name__)

Page = dict[str, Any]
TocEntry = list  # [level, title, page]

#: Pages sampled to decide whether the PDF has a text layer, and the text a page needs to count.
DIGITAL_SAMPLE = 20
DIGITAL_MIN_CHARS = 100
#: Blocks shorter than this are not paragraphs.
MIN_BLOCK_CHARS = 20
#: Pages scanned for a printed table of contents, and their resolution.
TOC_PAGES = 10
TOC_DPI = 150


def clean_whitespace(text: str) -> str:
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # hyphenated line breaks
    text = re.sub(r"\n", " ", text)
    return re.sub(r" +", " ", text).strip()


def parse_toc_line(text: str, x1: float) -> TocEntry | None:
    """``"Introduction ....... 12"`` at indent `x1` → ``[level, "Introduction", 12]``."""
    if not text:
        return None
    match = re.search(r"^(.*?)\s*[\.\s]{2,}\s*(\d+)\s*$", text) or re.search(r"^(.*?)\s+(\d+)\s*$", text)
    if not match:
        return None
    return [1 + int(x1 // 50), match.group(1).strip(), int(match.group(2))]


def _page(number: int, page: pymupdf.Page) -> Page:
    blocks = [clean_whitespace(block[4]) for block in page.get_text("blocks")]
    return {
        "page_num": number,
        "text": " ".join(blocks),
        "paragraphs": [block for block in blocks if len(block) > MIN_BLOCK_CHARS],
    }


class TocDetector:
    """A printed table of contents found by DocLayout-YOLO's `List-item` boxes."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _model(self):
        """The layout model; fetched to the Hugging Face cache when the image does not carry it."""
        from doclayout_yolo import YOLOv10

        path = self.settings.yolo_model_path
        if path is None or not path.exists():
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(repo_id=self.settings.yolo_repo_id, filename=self.settings.yolo_filename)
            except Exception as error:
                logger.warning("ToC layout model unavailable, skipping ToC detection: %s", error)
                return None
        return YOLOv10(str(path))

    @staticmethod
    def _list_items(model, image: Image.Image) -> list[list[float]]:
        result = model.predict(source=image, imgsz=1120, conf=0.2, iou=0.5, agnostic_nms=True, verbose=False)[0]
        boxes = [box.xyxy[0].tolist() for box in result.boxes if result.names[int(box.cls[0].item())] == "List-item"]
        return sorted(boxes, key=lambda box: box[1])

    def from_images(self, images: list[Image.Image]) -> list[TocEntry] | None:
        """The ToC of a scan: list items read with Tesseract on the first pages."""
        import pytesseract

        if (model := self._model()) is None:
            return None
        for image in images[:TOC_PAGES]:
            if not (items := self._list_items(model, image)):
                continue
            toc = []
            for x1, y1, x2, y2 in items:
                text = pytesseract.image_to_string(image.crop((x1, y1, x2, y2))).strip()
                if entry := parse_toc_line(text, x1):
                    toc.append(entry)
            return toc or None
        return None


class TextExtractor:
    """Every page's text and paragraphs, and the ToC; scans go through OCR first."""

    def __init__(self, settings: Settings, toc: TocDetector | None = None):
        self.toc = toc or TocDetector(settings)

    def extract(self, path: Path) -> tuple[list[Page], list[TocEntry] | None]:
        with pymupdf.open(path) as document:
            if self.is_digital(document):
                # A digital PDF's outline, as embedded (possibly empty); layout detection is for scans.
                pages = [_page(number, page) for number, page in enumerate(document, start=1)]
                return pages, document.get_toc()
        return self._extract_ocr(path)

    @staticmethod
    def is_digital(document: pymupdf.Document) -> bool:
        """Whether most sampled pages (from the middle 80% of the book) carry a text layer."""
        total = document.page_count
        population = list(range(int(total * 0.1), int(total * 0.9))) or list(range(total))
        if not population:
            return False
        sample = random.Random(42).sample(population, min(DIGITAL_SAMPLE, len(population)))
        digital = sum(
            1 for index in sample if len(document.load_page(index).get_text("text").strip()) > DIGITAL_MIN_CHARS
        )
        return digital / len(sample) >= 0.5

    def _extract_ocr(self, path: Path) -> tuple[list[Page], list[TocEntry] | None]:
        with pymupdf.open(stream=self._ocr(path), filetype="pdf") as document:
            images = []
            for index in range(min(TOC_PAGES, document.page_count)):
                pixmap = document.load_page(index).get_pixmap(dpi=TOC_DPI)
                images.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
            toc = self.toc.from_images(images)
            return [_page(number, page) for number, page in enumerate(document, start=1)], toc

    @staticmethod
    def _ocr(path: Path) -> bytes:
        """The PDF with an OCR text layer, in memory."""
        import ocrmypdf

        output = BytesIO()
        ocrmypdf.ocr(
            path,
            output,
            force_ocr=True,
            deskew=True,
            optimize=0,
            output_type="pdf",  # skip PDF/A generation
            fast_web_view=999999,  # skip fast web view optimization
        )
        return output.getvalue()
