import base64
import io
import json

import pymupdf
import pytest
import requests
from PIL import Image

from conftest import BODY, book
from evilflowers_text.extraction import decider, paragraphs, table_review, tables
from evilflowers_text.extraction.table_review import Review, TableReviewer
from evilflowers_worker import contract
from evilflowers_worker.contract import Box, LayoutRegion
from evilflowers_worker.judges import Budget, DeciderClient, Panel, Thresholds, Verdict


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
    assert paragraphs.embedding_text(first) == f"Information\n{first.text}"
    assert paragraphs.embedding_text(first.model_copy(update={"section": None})) == first.text


@pytest.mark.parametrize(
    ("outline", "sections"),
    [
        # Absent: the PDF's bookmarks, as before #15.
        (None, ["Information", "Information", "Coding", "Coding"]),
        # Present, even empty: authoritative.
        ([], [None, None, None, None]),
        ([(1, "Detected", 2)], [None, "Detected", "Detected", "Detected"]),
        ([[1, "One", 1], [2, "One.Two", 1], [1, "Three", 3]], ["One.Two", "One.Two", "Three", "Three"]),
    ],
)
def test_outline_from_the_analyzer_replaces_the_bookmarks(outline, sections):
    assert [row.section for row in paragraphs.extract(book(), min_words=10, outline=outline)] == sections


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
    reader = tables.TableReader(tmp_path, dpi=72, quality=50)
    box = Box(x0=0.15, y0=0.22, x1=0.8, y1=0.4)
    [region] = reader.read(ruled_table_pdf, "runs/1", [LayoutRegion(page=1, seq=3, bbox=box, caption="Table 1")])
    assert region.kind == "table" and region.image_path == "runs/1/p001-tab03.webp"
    assert (tmp_path / region.image_path).stat().st_size > 0
    assert region.printed_caption == region.caption == "Table 1"
    assert region.markdown.startswith("| R0C0 | R0C1 | R0C2 |")


def test_table_reader_skips_a_bad_region(tmp_path, ruled_table_pdf):
    reader = tables.TableReader(tmp_path, dpi=72, quality=50)
    box = Box(x0=0.1, y0=0.1, x1=0.5, y1=0.5)
    assert reader.read(ruled_table_pdf, "runs/1", [LayoutRegion(page=9, seq=0, bbox=box)]) == []


DRAFT = "| A | B |\n| --- | --- |\n| 1 | 2 |"
REWRITE = "| A | B |\n| --- | --- |\n| 1 | 3 |"


@pytest.mark.parametrize(
    ("content", "draft", "expected"),
    [
        ('{"ok": true}', DRAFT, Review(DRAFT, "kept")),
        (json.dumps({"ok": False, "markdown": REWRITE}), DRAFT, Review(REWRITE, "rewritten")),
        (json.dumps({"ok": False, "markdown": REWRITE}), None, Review(REWRITE, "transcribed")),
        # `ok` with no draft to keep: the answer has no table, nothing is transcribed.
        ('{"ok": true}', None, Review(None, "unusable")),
        ("not json", DRAFT, Review(DRAFT, "unusable")),
        ("[1, 2]", DRAFT, Review(DRAFT, "unusable")),
        ('{"ok": false, "markdown": "| a |"}', DRAFT, Review(DRAFT, "unusable")),
    ],
)
def test_review_says_what_it_did(content, draft, expected):
    assert table_review.parse(content, draft) == expected


def test_a_failed_review_keeps_the_draft(monkeypatch, caplog):
    def refuse(*args, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr(table_review.requests, "post", refuse)
    reviewer = TableReviewer("http://ollama", "m", api_key=None, timeout=1, max_side=64)
    assert reviewer.review(Image.new("RGB", (8, 8)), DRAFT) == Review(DRAFT, "failed")
    assert "Table review failed" in caplog.text


def test_grounded_share_finds_the_values_in_the_pdf_text():
    text = "Year Regression coef-\nficient\n2012 0.45***\n2014 (1,234.56)"
    table = "| Year | Regression coefficient |\n| --- | --- |\n| 2012 | 0.45*** |\n| 2014 | (1,234.56) |"
    assert tables.grounded_share(table, text) == 1.0
    # An invented number is not found, even inside another one (12 in 2012).
    invented = table.replace("0.45", "12")
    assert tables.grounded_share(invented, text) == pytest.approx(6 / 7, abs=0.001)
    assert tables.grounded_share(table.replace("1,234.56", "1,234.57"), text) < 1.0


def test_grounded_share_is_unknown_without_markdown_or_text_layer():
    assert tables.grounded_share(None, "2012") is None
    assert tables.grounded_share(DRAFT, "") is None
    assert tables.grounded_share("| --- |", "2012") is None


def test_a_rotated_table_has_no_grounded_share(tmp_path):
    # pdfplumber reads rotated text reversed ("tneiciffeoC"): a correct transcription would score 0.
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)
    for row, line in enumerate(["Year   Coefficient", "2012   0.45", "2014   1,234.56"]):
        page.insert_text((200 + row * 20, 600), line, fontsize=10, rotate=90)
    path = tmp_path / "rotated.pdf"
    pdf.save(path)
    transcription = "| Year | Coefficient |\n| --- | --- |\n| 2012 | 0.45 |\n| 2014 | 1,234.56 |"
    reader = tables.TableReader(
        tmp_path, dpi=72, quality=50, reviewer=FakeReviewer(Review(transcription, "transcribed"))
    )
    [region] = reader.read(path, "runs/1", [LayoutRegion(page=1, seq=0, bbox=Box(x0=0.2, y0=0.4, x1=0.6, y1=0.8))])
    assert region.markdown == transcription and region.provenance["review_outcome"] == "transcribed"
    assert region.provenance["grounded_share"] is None
    assert region.provenance["rotated_share"] == 1.0 and region.provenance["grid_strategy"] is None


class FakeReviewer:
    model = "fake"

    def __init__(self, review):
        self.result = review

    def review(self, image, draft):
        return self.result if self.result.outcome != "kept" else Review(draft, "kept")


def test_table_region_carries_its_quality_signals(tmp_path, ruled_table_pdf):
    box = Box(x0=0.15, y0=0.22, x1=0.8, y1=0.4)
    regions = [LayoutRegion(page=1, seq=3, bbox=box)]

    [plain] = tables.TableReader(tmp_path, dpi=72, quality=50).read(ruled_table_pdf, "runs/1", regions)
    assert plain.provenance["grid_strategy"] == "lines"
    assert plain.provenance["grid_filled_share"] == 1.0
    assert plain.provenance["grid_fragment_share"] == 0.0
    assert plain.provenance["grounded_share"] == 1.0
    assert "review_outcome" not in plain.provenance

    reviewer = FakeReviewer(Review(None, "kept"))
    [kept] = tables.TableReader(tmp_path, dpi=72, quality=50, reviewer=reviewer).read(
        ruled_table_pdf, "runs/1", regions
    )
    assert kept.provenance["review_outcome"] == "kept" and kept.markdown == plain.markdown

    invented = "| R0C0 | R0C1 | R0C2 |\n| --- | --- | --- |\n| 9999 | 8888 | 7777 |"
    reviewer = FakeReviewer(Review(invented, "rewritten"))
    [rewritten] = tables.TableReader(tmp_path, dpi=72, quality=50, reviewer=reviewer).read(
        ruled_table_pdf, "runs/1", regions
    )
    assert rewritten.provenance["review_outcome"] == "rewritten"
    assert rewritten.provenance["grounded_share"] < 1.0


# -- The decider: "is this a data table?" (issue #20) ------------------------------------------------------------

DECISION_FIELDS = {"band", "p", "score", "decision", "decider_model", "review_outcome", "grounded_share", "signals"}
TABLE_BOX = Box(x0=0.15, y0=0.22, x1=0.8, y1=0.4)


class FakeDecider:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    def is_table(self, image):
        self.calls += 1
        return self.verdict


class CountingReviewer(FakeReviewer):
    def __init__(self, review):
        super().__init__(review)
        self.calls = 0

    def review(self, image, draft):
        self.calls += 1
        return super().review(image, draft)


def read_one(tmp_path, pdf, **kwargs):
    [region] = tables.TableReader(tmp_path, dpi=72, quality=50, **kwargs).read(
        pdf, "runs/1", [LayoutRegion(page=1, seq=3, bbox=TABLE_BOX)]
    )
    return contract.dump(region)


def verdict(answer, p, band=None, model="strands"):
    return Verdict(question="is_table", verdict=answer, p=p, model=model, revision="216d1ccb", band=band)


class FakeHttp:
    """The kit client's `requests.Session`: records the call, answers `body` (or raises `error`)."""

    def __init__(self, body=None, error=None):
        self.body, self.error, self.sent = body, error, {}

    def post(self, url, **kwargs):
        self.sent.update(url=url, **kwargs)
        if self.error:
            raise self.error
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(self.body).encode()
        return response


def table_decider(http, thresholds=None):
    client = DeciderClient("decider-vision", "http://dv/v1/systemone", vision=True, timeout=7, http=http)
    return decider.TableDecider(client, Panel(Budget(10, 0), thresholds or {}))


def test_decider_asks_the_kits_judge_with_the_crop_and_the_noul_question():
    http = FakeHttp({"answers": {"is_table": {"noul": 0.9}}, "model": "strands"})
    answer = table_decider(http).is_table(Image.new("RGB", (2000, 1000), "white"))
    assert answer.verdict == "yes" and answer.p == pytest.approx(0.9) and answer.band is None
    assert http.sent["url"] == "http://dv/v1/systemone" and http.sent["timeout"] == 7
    question = http.sent["json"]["questions"]["is_table"]
    assert question["type"] == "noul" and "table of contents" in question["instructions"]
    [image] = http.sent["json"]["images"]
    crop = Image.open(io.BytesIO(base64.b64decode(image)))
    assert crop.format == "WEBP" and max(crop.size) == decider.MAX_SIDE


def test_the_kits_thresholds_band_a_non_table_hide():
    http = FakeHttp({"answers": {"is_table": {"noul": 0.05}}, "model": "strands"})
    answer = table_decider(http, {"is_table": Thresholds(keep=0.8, hide=0.9)}).is_table(Image.new("RGB", (8, 8)))
    assert answer.verdict == "no" and answer.p == pytest.approx(0.95) and answer.band == "hide"
    assert decider.p_table(answer) == pytest.approx(0.05)


@pytest.mark.parametrize(
    "http",
    [
        FakeHttp(error=requests.ConnectionError("down")),
        # Never read an odd answer as p = 0: it is no answer.
        FakeHttp({"answers": {"is_table": {"choice": "maybe"}}}),
        FakeHttp({"detail": "Not found"}),
    ],
)
def test_no_answer_is_no_verdict_never_an_error(http):
    assert table_decider(http).is_table(Image.new("RGB", (8, 8))) is None


def test_without_a_verdict_the_payload_is_the_one_before_the_decider(tmp_path, ruled_table_pdf):
    for kwargs in ({}, {"decider": FakeDecider(None)}):
        payload = read_one(tmp_path, ruled_table_pdf, **kwargs)
        assert not DECISION_FIELDS & payload.keys()
        assert payload["provenance"]["grounded_share"] == 1.0


def test_a_data_table_is_reviewed_scored_and_banded(tmp_path, ruled_table_pdf):
    reviewer = CountingReviewer(Review(None, "kept"))
    payload = read_one(
        tmp_path,
        ruled_table_pdf,
        reviewer=reviewer,
        decider=FakeDecider(verdict("yes", 0.9, band="keep")),
        decision=tables.Decision(bands=(0.4, 0.7)),
    )
    assert reviewer.calls == 1
    assert payload["band"] == "keep" and payload["p"] == 0.9 and payload["decision"] == "yes"
    assert payload["decider_model"] == "strands@216d1ccb"
    # Grounded 1.0 (×3), grid 1.0 (×2), kept 1.0 (×1), p 0.9 (×1).
    assert payload["score"] == pytest.approx((3 + 2 + 1 + 0.9) / 7, abs=0.001)
    assert payload["review_outcome"] == "kept" and payload["grounded_share"] == 1.0
    assert payload["signals"] == {
        "filled_share": 1.0,
        "fragment_share": 0.0,
        "strategy": "lines",
        "rotated_share": 0.0,
    }
    # The #19 provenance stays as it was.
    assert payload["provenance"]["review_outcome"] == "kept" and payload["provenance"]["grid_strategy"] == "lines"


def test_not_a_data_table_skips_the_review_and_is_hidden(tmp_path, ruled_table_pdf):
    reviewer = CountingReviewer(Review(None, "kept"))
    payload = read_one(
        tmp_path,
        ruled_table_pdf,
        reviewer=reviewer,
        decider=FakeDecider(verdict("no", 0.95, band="hide")),
        decision=tables.Decision(bands=(0.4, 0.7)),
    )
    assert reviewer.calls == 0
    assert payload["band"] == "hide" and payload["p"] == 0.95 and payload["decision"] == "no"
    # A table of contents is perfectly grounded: no score, so the catalog cannot re-band it upward.
    assert "score" not in payload and "review_outcome" not in payload
    # The crop and the draft are kept; the region is still posted.
    assert payload["image_path"] == "runs/1/p001-tab03.webp" and payload["markdown"].startswith("| R0C0")


def test_without_thresholds_the_region_has_p_and_score_but_no_band(tmp_path, ruled_table_pdf):
    reviewer = CountingReviewer(Review(None, "kept"))
    # A confident "no" with no `is_table` thresholds: not banded by the kit, so not skipped.
    payload = read_one(tmp_path, ruled_table_pdf, reviewer=reviewer, decider=FakeDecider(verdict("no", 1.0)))
    assert reviewer.calls == 1
    assert "band" not in payload and payload["p"] == 1.0 and payload["decision"] == "no"
    # p of "a data table" is 0: grounded 1.0 (×3), grid 1.0 (×2), kept 1.0 (×1), 0 (×1).
    assert payload["score"] == pytest.approx(6 / 7, abs=0.001)


def test_a_kept_or_review_band_of_the_answer_does_not_skip(tmp_path, ruled_table_pdf):
    reviewer = CountingReviewer(Review(None, "kept"))
    payload = read_one(tmp_path, ruled_table_pdf, reviewer=reviewer, decider=FakeDecider(verdict("no", 0.6, "review")))
    assert reviewer.calls == 1 and "band" not in payload and payload["score"] is not None


@pytest.mark.parametrize(
    ("score", "band"), [(None, None), (0.1, "hide"), (0.4, "review"), (0.69, "review"), (0.7, "keep")]
)
def test_bands_from_the_score(score, band):
    assert tables.Decision(bands=(0.4, 0.7)).band(score) == band
    assert tables.Decision().band(score) is None


def test_table_score_leaves_out_what_is_unknown():
    no_grid = tables.Grid(None)
    assert tables.table_score(None, no_grid, None, None) is None
    assert tables.table_score(None, no_grid, None, 0.3) == 0.3
    # An implausible grid counts as 0; a failed review says nothing about the table.
    implausible = tables.Grid(None, None, 0.2, 0.6)
    assert tables.table_score(0.9, implausible, "failed", None) == pytest.approx(3 * 0.9 / 5, abs=0.001)
    assert tables.table_score(1.0, tables.Grid("|", "lines", 0.8, 0.25), "rewritten", 1.0) == pytest.approx(
        (3 + 2 * 0.6 + 0.5 + 1) / 7, abs=0.001
    )
