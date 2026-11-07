import pymupdf
import pytest

from evilflowers_text.settings import Settings

BODY = (
    "Entropy measures the uncertainty of a random variable. It is defined over the distribution, "
    "e.g. a fair coin has one bit. Napr. kocka má viac stavov a teda aj vyššiu entropiu."
)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Defaults only (no `.env`), storage in a temporary directory, no retry sleeps worth waiting for."""
    return Settings(_env_file=None, storage_path=tmp_path, catalog_attempts=2, image_tag="test")


def book(pages: int = 4) -> pymupdf.Document:
    """A small book: a running head, one paragraph and a page number per page, and a two-entry outline."""
    pdf = pymupdf.open()
    for number in range(1, pages + 1):
        page = pdf.new_page(width=595, height=842)
        page.insert_text((60, 40), "Running Head of the Book", fontsize=9)
        page.insert_textbox(pymupdf.Rect(60, 120, 535, 400), BODY, fontsize=11)
        page.insert_text((290, 820), str(number), fontsize=9)
    pdf.set_toc([[1, "Information", 1], [1, "Coding", 3]])
    return pdf
