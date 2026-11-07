"""The parts of the search-service pipeline that need no OCR, layout model or llama-index."""

import pymupdf
import pytest

from conftest import book
from evilflowers_text.indexing import chunker
from evilflowers_text.indexing.tables import table_extractor
from evilflowers_text.indexing.text_extractor import TextExtractor, parse_toc_line


def test_toc_line_with_leaders():
    assert parse_toc_line("Introduction ........ 12", x1=120) == [3, "Introduction", 12]
    assert parse_toc_line("no page number", x1=0) is None


def test_page_offsets_and_source_page():
    pages = [{"page_num": 1, "text": "abc"}, {"page_num": 2, "text": ""}, {"page_num": 3, "text": "de"}]
    offsets = chunker.page_offsets(pages)
    assert offsets == [(0, 4, 1), (4, 7, 3)]
    assert chunker.source_page(5, offsets) == 3
    assert chunker.source_page(99, offsets) == 1


def test_section_is_the_last_heading_on_or_before_the_page():
    toc = [[1, "One", 1], [2, "One.A", 3], [1, "Two", 7]]
    assert chunker.section(4, toc) == "One.A"
    assert chunker.section(1, None) is None


@pytest.mark.parametrize("pages", [1, 2, 4])
def test_short_digital_books_are_digital(pages):
    # A one-page PDF used to divide by an empty sample.
    assert TextExtractor.is_digital(book(pages)) is True


def test_blank_pages_are_not_digital():
    pdf = pymupdf.open()
    pdf.new_page()
    assert TextExtractor.is_digital(pdf) is False


def test_unknown_table_extractor_is_named():
    with pytest.raises(ValueError, match="PdfPlumber"):
        table_extractor("Tabula")
