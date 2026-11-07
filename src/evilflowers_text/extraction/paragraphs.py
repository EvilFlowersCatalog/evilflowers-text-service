"""Paragraph index of an extraction run (IP-018 Phase 2b).

Paragraphs come from the PDF's text layer (PyMuPDF words grouped by block),
not from layout, so this job runs in parallel with figures, tables and
equations. Each paragraph keeps its page and box, its sentences as character
offsets with their quads, and the outline heading it falls under. Running
heads and footers, page numbers, short fragments and table-of-contents
boilerplate are dropped (tp2025 `_is_valid_paragraph`).
"""

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import pymupdf

from evilflowers_text.extraction.geometry import to_box
from evilflowers_text.schemas import Paragraph, ParagraphLink, Sentence

#: tp2025 `_is_valid_paragraph` boilerplate.
BOILERPLATE = ("table of contents", "obsah", "zoznam obrázkov", "zoznam tabuliek")

#: Words ending in a period that do not end a sentence.
ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "et al.", "vs.", "cf.", "fig.", "figs.", "eq.", "eqs.", "no.", "vol.", "pp.",
    "p.", "dr.", "prof.", "mr.", "mrs.", "ms.", "st.", "napr.", "tzv.", "tj.", "t.j.", "resp.", "obr.",
    "tab.", "str.", "s.", "č.", "roč.", "zv.", "ing.", "doc.", "mgr.", "bc.", "phd.", "rnd.", "atď.",
}  # fmt: skip

_SENTENCE_END = re.compile(r"[.!?…]+[\"'»”’)\]]*$")
_STARTS_SENTENCE = re.compile(r"^[\"'„“«(\[]*[A-ZÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ0-9]")
_DIGITS = re.compile(r"\d+")

#: Share of the page height at the top and bottom where running heads live.
MARGIN = 0.1
#: Text in the margins of at least this share of pages (and at least 3) is a running head or footer.
RUNNING_SHARE = 0.3
#: Rows of the similarity matrix computed at once (bounds memory on long books).
SIMILARITY_BLOCK = 512


@dataclass
class _Word:
    start: int
    end: int
    rect: pymupdf.Rect
    line: tuple[int, int]


@dataclass
class _Block:
    page: int
    text: str
    rect: pymupdf.Rect
    words: list[_Word] = field(default_factory=list)

    @property
    def running_key(self) -> str:
        return _DIGITS.sub("#", self.text.lower())[:80]

    def in_margin(self, page_height: float) -> bool:
        return self.rect.y1 <= page_height * MARGIN or self.rect.y0 >= page_height * (1 - MARGIN)

    def quads(self, start: int, end: int, width: float, height: float) -> list:
        """One box per line of the words a sentence covers, as page fractions."""
        lines: dict[tuple[int, int], pymupdf.Rect] = {}
        for word in self.words:
            if word.end > start and word.start < end:
                lines[word.line] = lines[word.line] | word.rect if word.line in lines else pymupdf.Rect(word.rect)
        return [to_box(rect, width, height) for rect in lines.values()]


def _blocks(page: pymupdf.Page, page_number: int) -> list[_Block]:
    by_block: dict[int, list] = defaultdict(list)
    for x0, y0, x1, y1, text, block, line, number in page.get_text("words"):
        by_block[block].append((line, number, text, pymupdf.Rect(x0, y0, x1, y1)))

    blocks = []
    for block in sorted(by_block):
        parts: list[str] = []
        words: list[_Word] = []
        length = 0
        previous_line = None
        for line, _number, text, rect in sorted(by_block[block], key=lambda word: (word[0], word[1])):
            joined = False
            if parts:
                # A hyphen at the end of a line joins the word across the break.
                if line != previous_line and parts[-1].endswith("-") and len(parts[-1]) > 1 and text[:1].islower():
                    parts[-1] = parts[-1][:-1]
                    length -= 1
                    words[-1].end -= 1
                    joined = True
                else:
                    parts.append(" ")
                    length += 1
            start = length
            parts.append(text)
            length += len(text)
            words.append(_Word(start, length, rect, (block, line)))
            if joined:
                words[-1].start = words[-2].start if len(words) > 1 else start
            previous_line = line
        text = "".join(parts).strip()
        if text:
            box = pymupdf.Rect(words[0].rect)
            for word in words[1:]:
                box |= word.rect
            blocks.append(_Block(page_number, text, box, words))
    return blocks


def is_valid(text: str, min_words: int) -> bool:
    if len(text.split()) < min_words:
        return False
    lower = text.lower()
    return not any(pattern in lower for pattern in BOILERPLATE)


def sentences(text: str) -> list[tuple[int, int]]:
    """Character offsets of the sentences of `text`."""
    spans = []
    start = 0
    tokens = list(re.finditer(r"\S+", text))
    for index, token in enumerate(tokens):
        word = token.group()
        if index == len(tokens) - 1:
            spans.append((start, token.end()))
            break
        if not _SENTENCE_END.search(word) or word.lower() in ABBREVIATIONS:
            continue
        if len(word) == 2 and word[0].isupper():  # an initial: "J. Smith"
            continue
        if not _STARTS_SENTENCE.match(tokens[index + 1].group()):
            continue
        spans.append((start, token.end()))
        start = tokens[index + 1].start()
    return [(begin, end) for begin, end in spans if text[begin:end].strip()]


def _sections(toc: list, page_count: int) -> list[str | None]:
    """The innermost outline heading in force on each page (index 0 is page 1)."""
    headings = sorted((page, order, title.strip()) for order, (_level, title, page, *_rest) in enumerate(toc))
    by_page: list[str | None] = []
    current = None
    position = 0
    for page in range(1, page_count + 1):
        while position < len(headings) and headings[position][0] <= page:
            if headings[position][2]:
                current = headings[position][2]
            position += 1
        by_page.append(current)
    return by_page


def _running_heads(pages: list[list[_Block]], document: pymupdf.Document) -> set[str]:
    """Text repeated in the margins of many pages: a running head or footer."""
    keys: Counter[str] = Counter()
    for index, blocks in enumerate(pages):
        height = document[index].rect.height
        keys.update({block.running_key for block in blocks if block.in_margin(height)})
    threshold = max(3, int(len(pages) * RUNNING_SHARE))
    return {key for key, count in keys.items() if count >= threshold}


def extract(document: pymupdf.Document, min_words: int) -> list[Paragraph]:
    """Every paragraph of the document, in reading order, as the catalog ingests it (without embeddings)."""
    pages = [_blocks(page, index + 1) for index, page in enumerate(document)]
    running = _running_heads(pages, document)
    sections = _sections(document.get_toc(simple=True), len(pages))

    result: list[Paragraph] = []
    for index, blocks in enumerate(pages):
        width, height = document[index].rect.width, document[index].rect.height
        for block in blocks:
            if block.in_margin(height) and block.running_key in running:
                continue
            if not is_valid(block.text, min_words):
                continue
            result.append(
                Paragraph(
                    page=block.page,
                    seq=len(result),
                    section=sections[index],
                    bbox=to_box(block.rect, width, height),
                    text=block.text,
                    sentences=[
                        Sentence(
                            seq=number,
                            start=start,
                            end=end,
                            page=block.page,
                            quads=block.quads(start, end, width, height),
                        )
                        for number, (start, end) in enumerate(sentences(block.text))
                    ],
                )
            )
    return result


def links(embeddings: Sequence[Sequence[float]], k: int, floor: float) -> list[ParagraphLink]:
    """`next` in reading order, and `similar` to the k nearest other paragraphs above `floor`."""
    import numpy as np

    count = len(embeddings)
    result = [ParagraphLink(source=index, target=index + 1, kind="next") for index in range(count - 1)]
    if count < 2 or k <= 0:
        return result

    vectors = np.asarray(embeddings, dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    for first in range(0, count, SIMILARITY_BLOCK):
        scores = vectors[first : first + SIMILARITY_BLOCK] @ vectors.T
        for row, source in enumerate(range(first, min(first + SIMILARITY_BLOCK, count))):
            scores[row, source] = -1.0
            for target in np.argsort(-scores[row])[:k]:
                score = float(scores[row, target])
                if score < floor:
                    break
                result.append(ParagraphLink(source=source, target=int(target), kind="similar", score=round(score, 4)))
    return result
