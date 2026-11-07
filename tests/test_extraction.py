import pymupdf
import pytest

from conftest import BODY, book
from evilflowers_text.extraction import paragraphs, tables


def test_sentences_respect_abbreviations():
    spans = paragraphs.sentences(BODY)
    texts = [BODY[start:end] for start, end in spans]
    assert texts == [
        "Entropy measures the uncertainty of a random variable.",
        "It is defined over the distribution, e.g. a fair coin has one bit.",
        "Napr. kocka má viac stavov a teda aj vyššiu entropiu.",
    ]


def test_paragraphs_drop_running_heads_and_keep_positions():
    rows = paragraphs.extract(book(), min_words=10)
    assert len(rows) == 4
    assert all("Running Head" not in row.text for row in rows)
    first = rows[0]
    assert first.seq == 0 and first.page == 1 and first.section == "Information"
    assert rows[2].section == "Coding"
    assert first.text.startswith("Entropy measures")
    assert len(first.sentences) == 3
    for sentence in first.sentences:
        assert first.text[sentence.start : sentence.end].strip()
        assert sentence.quads and all(0 <= quad.x0 < quad.x1 <= 1 for quad in sentence.quads)
    assert 0 < first.bbox.y0 < first.bbox.y1 < 0.5


def test_paragraph_payload_is_what_the_catalog_ingests():
    payload = paragraphs.extract(book(), min_words=10)[0].model_dump(mode="json", exclude_none=True)
    assert set(payload) == {"page", "seq", "section", "bbox", "text", "sentences"}
    assert set(payload["bbox"]) == {"x0", "y0", "x1", "y1"}
    assert set(payload["sentences"][0]) == {"seq", "start", "end", "page", "quads"}


def test_embedding_text_prefixes_the_section():
    first = paragraphs.extract(book(), min_words=10)[0]
    assert first.embedding_text == f"Information\n{first.text}"


def test_short_paragraphs_are_dropped():
    assert paragraphs.extract(book(), min_words=100) == []


def test_hyphenated_line_break_is_joined():
    pdf = pymupdf.open()
    page = pdf.new_page()
    text = (
        "The receiver counted the seated people from the vari-\nance of the channel amplitude over a long time window."
    )
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 300), text, fontsize=11)
    rows = paragraphs.extract(pdf, min_words=10)
    assert "variance" in rows[0].text


def test_links_next_and_similar():
    vectors = [[1, 0], [0.99, 0.1], [0, 1]]
    result = paragraphs.links(vectors, k=1, floor=0.5)
    assert ("next", 0, 1) in {(link.kind, link.source, link.target) for link in result}
    similar = [link for link in result if link.kind == "similar"]
    assert {(link.source, link.target) for link in similar} == {(0, 1), (1, 0)}


def test_markdown_table():
    assert tables.markdown([["A", "B"], ["1", None], ["x|y", "2"]]) == (
        "| A | B |\n| --- | --- |\n| 1 |  |\n| x\\|y | 2 |"
    )
    assert tables.markdown([[None, ""]]) is None


def test_garbled_tables_get_no_markdown():
    # The landscape Table 5.1 read sideways: letters scattered over cells.
    garbled = [
        ["", "", "", "a fo", ""],
        ["", "", "", "regan", "tc"],
        ["dnuo", "", "", "orgk", "t c"],
        ["rgk", "ro", "ro", "cab", "yt"],
        ["c", "od", "od", "me uf", ",y"],
    ]
    assert not tables.plausible(garbled)
    real = [["Year", "Event"], ["1977", "Health Council report"], ["2014", "Ban on tobacco sales to minors"]]
    assert tables.plausible(real)


def test_rotated_text_is_detected():
    assert tables.rotated_share([{"upright": False}] * 8 + [{"upright": True}] * 2) > tables.MAX_ROTATED_SHARE
    assert tables.rotated_share([{"upright": True}] * 10) == 0


@pytest.fixture
def ruled_table_pdf(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)
    for row in range(4):
        for column in range(3):
            rect = pymupdf.Rect(100 + column * 120, 200 + row * 30, 220 + column * 120, 230 + row * 30)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 5, rect.y0 + 20), f"R{row}C{column}", fontsize=10)
    path = tmp_path / "catalogs" / "book.pdf"
    path.parent.mkdir(parents=True)
    pdf.save(path)
    return path


def test_table_reader_crops_and_reads_a_ruled_table(tmp_path, ruled_table_pdf):
    from evilflowers_text.schemas import Box, LayoutRegion

    reader = tables.TableReader(tmp_path, dpi=72, quality=50)
    box = Box(x0=0.15, y0=0.22, x1=0.8, y1=0.4)
    [region] = reader.read(ruled_table_pdf, "runs/1", [LayoutRegion(page=1, seq=3, bbox=box, caption="Table 1")])
    assert region.kind == "table" and region.image_path == "runs/1/p001-tab03.webp"
    assert (tmp_path / region.image_path).stat().st_size > 0
    assert region.printed_caption == region.caption == "Table 1"
    assert region.markdown.startswith("| R0C0 | R0C1 | R0C2 |")


def test_table_reader_skips_a_bad_region(tmp_path, ruled_table_pdf):
    from evilflowers_text.schemas import Box, LayoutRegion

    reader = tables.TableReader(tmp_path, dpi=72, quality=50)
    box = Box(x0=0.1, y0=0.1, x1=0.5, y1=0.5)
    assert reader.read(ruled_table_pdf, "runs/1", [LayoutRegion(page=9, seq=0, bbox=box)]) == []
