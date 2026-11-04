import fitz
import pytest

import paragraphs
import tables

BODY = (
    "Entropy measures the uncertainty of a random variable. It is defined over the distribution, "
    "e.g. a fair coin has one bit. Napr. kocka má viac stavov a teda aj vyššiu entropiu."
)


def _document(pages=4):
    pdf = fitz.open()
    for number in range(1, pages + 1):
        page = pdf.new_page(width=595, height=842)
        page.insert_text((60, 40), "Running Head of the Book", fontsize=9)
        page.insert_textbox(fitz.Rect(60, 120, 535, 400), BODY, fontsize=11)
        page.insert_text((290, 820), str(number), fontsize=9)
    pdf.set_toc([[1, "Information", 1], [1, "Coding", 3]])
    return pdf


def test_sentences_respect_abbreviations():
    spans = paragraphs.sentences(BODY)
    texts = [BODY[start:end] for start, end in spans]
    assert texts == [
        "Entropy measures the uncertainty of a random variable.",
        "It is defined over the distribution, e.g. a fair coin has one bit.",
        "Napr. kocka má viac stavov a teda aj vyššiu entropiu.",
    ]


def test_paragraphs_drop_running_heads_and_keep_positions():
    pdf = _document()
    rows = paragraphs.extract(pdf)
    assert len(rows) == 4
    assert all("Running Head" not in row["text"] for row in rows)
    first = rows[0]
    assert first["seq"] == 0 and first["page"] == 1 and first["section"] == "Information"
    assert rows[2]["section"] == "Coding"
    assert first["text"].startswith("Entropy measures")
    assert len(first["sentences"]) == 3
    for sentence in first["sentences"]:
        assert first["text"][sentence["start"] : sentence["end"]].strip()
        assert sentence["quads"] and all(0 <= quad["x0"] < quad["x1"] <= 1 for quad in sentence["quads"])
    assert 0 < first["bbox"]["y0"] < first["bbox"]["y1"] < 0.5


def test_hyphenated_line_break_is_joined():
    pdf = fitz.open()
    page = pdf.new_page()
    text = "The receiver counted the seated people from the vari-\nance of the channel amplitude over a long time window."
    page.insert_textbox(fitz.Rect(50, 50, 550, 300), text, fontsize=11)
    rows = paragraphs.extract(pdf)
    assert "variance" in rows[0]["text"]


def test_links_next_and_similar():
    numpy = pytest.importorskip("numpy")
    vectors = numpy.array([[1, 0], [0.99, 0.1], [0, 1]], dtype=float).tolist()
    result = paragraphs.links(vectors, k=1, floor=0.5)
    assert {"source": 0, "target": 1, "kind": "next"} in result
    similar = [link for link in result if link["kind"] == "similar"]
    assert {(link["source"], link["target"]) for link in similar} == {(0, 1), (1, 0)}


def test_markdown_table():
    assert tables.markdown([["A", "B"], ["1", None], ["x|y", "2"]]) == (
        "| A | B |\n| --- | --- |\n| 1 |  |\n| x\\|y | 2 |"
    )
    assert tables.markdown([[None, ""]]) is None


def test_no_paragraphs_posts_nothing(monkeypatch):
    from catalog import Catalog

    client = Catalog("run")
    monkeypatch.setattr(client, "_post", lambda *args, **kwargs: pytest.fail("posted an empty batch"))
    client.paragraphs([], [])
