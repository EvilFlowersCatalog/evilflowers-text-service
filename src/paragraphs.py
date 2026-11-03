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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import fitz

import settings

#: tp2025 `_is_valid_paragraph` boilerplate.
BOILERPLATE = ("table of contents", "obsah", "zoznam obrázkov", "zoznam tabuliek")

#: Words ending in a period that do not end a sentence.
ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "et al.", "vs.", "cf.", "fig.", "figs.", "eq.", "eqs.", "no.", "vol.", "pp.",
    "p.", "dr.", "prof.", "mr.", "mrs.", "ms.", "st.", "napr.", "tzv.", "tj.", "t.j.", "resp.", "obr.",
    "tab.", "str.", "s.", "č.", "roč.", "zv.", "ing.", "doc.", "mgr.", "bc.", "phd.", "rnd.", "atď.",
}

_SENTENCE_END = re.compile(r"[.!?…]+[\"'»”’)\]]*$")
_STARTS_SENTENCE = re.compile(r"^[\"'„“«(\[]*[A-ZÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ0-9]")
_DIGITS = re.compile(r"\d+")

#: Share of the page height at the top and bottom where running heads live.
MARGIN = 0.1


@dataclass
class Word:
    start: int
    end: int
    rect: fitz.Rect
    line: Tuple[int, int]


@dataclass
class Paragraph:
    page: int
    text: str
    rect: fitz.Rect
    words: List[Word] = field(default_factory=list)


def _block_paragraphs(page: fitz.Page, page_number: int) -> List[Paragraph]:
    blocks: Dict[int, list] = defaultdict(list)
    for x0, y0, x1, y1, text, block, line, number in page.get_text("words"):
        blocks[block].append((line, number, text, fitz.Rect(x0, y0, x1, y1)))

    paragraphs = []
    for block in sorted(blocks):
        parts: List[str] = []
        words: List[Word] = []
        length = 0
        previous_line = None
        for line, _number, text, rect in sorted(blocks[block], key=lambda word: (word[0], word[1])):
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
            words.append(Word(start, length, rect, (block, line)))
            if joined:
                words[-1].start = words[-2].start if len(words) > 1 else start
            previous_line = line
        text = "".join(parts).strip()
        if text:
            box = fitz.Rect(words[0].rect)
            for word in words[1:]:
                box |= word.rect
            paragraphs.append(Paragraph(page_number, text, box, words))
    return paragraphs


def _running_key(paragraph: Paragraph) -> str:
    return _DIGITS.sub("#", paragraph.text.lower())[:80]


def _in_margin(paragraph: Paragraph, page_height: float) -> bool:
    return paragraph.rect.y1 <= page_height * MARGIN or paragraph.rect.y0 >= page_height * (1 - MARGIN)


def is_valid(text: str) -> bool:
    if len(text.split()) < settings.MIN_PARAGRAPH_WORDS:
        return False
    lower = text.lower()
    return not any(pattern in lower for pattern in BOILERPLATE)


def sentences(text: str) -> List[Tuple[int, int]]:
    """Character offsets of the sentences of `text`."""
    spans = []
    start = 0
    tokens = list(re.finditer(r"\S+", text))
    for index, token in enumerate(tokens):
        word = token.group()
        last = index == len(tokens) - 1
        if last:
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


def _quads(paragraph: Paragraph, start: int, end: int, width: float, height: float) -> List[dict]:
    """One box per line of the words a sentence covers, as page fractions."""
    lines: Dict[Tuple[int, int], fitz.Rect] = {}
    for word in paragraph.words:
        if word.end > start and word.start < end:
            lines[word.line] = lines[word.line] | word.rect if word.line in lines else fitz.Rect(word.rect)
    return [_fractions(rect, width, height) for rect in lines.values()]


def _fractions(rect: fitz.Rect, width: float, height: float) -> dict:
    def clamp(value: float) -> float:
        return round(min(max(value, 0.0), 1.0), 5)

    return {
        "x0": clamp(rect.x0 / width),
        "y0": clamp(rect.y0 / height),
        "x1": clamp(rect.x1 / width),
        "y1": clamp(rect.y1 / height),
    }


def _sections(toc: list, page_count: int) -> List[Optional[str]]:
    """The innermost outline heading in force on each page (index 0 is page 1)."""
    by_page: List[Optional[str]] = [None] * (page_count + 1)
    headings = sorted(((page, order, title.strip()) for order, (_level, title, page, *_rest) in enumerate(toc)))
    current = None
    position = 0
    for page in range(1, page_count + 1):
        while position < len(headings) and headings[position][0] <= page:
            if headings[position][2]:
                current = headings[position][2]
            position += 1
        by_page[page] = current
    return by_page[1:]


def extract(document: fitz.Document) -> List[dict]:
    """Every paragraph of the document, in reading order, as the catalog ingests it (without embeddings)."""
    pages = [_block_paragraphs(page, index + 1) for index, page in enumerate(document)]

    # Text repeated in the margins of many pages is a running head or footer.
    keys = Counter()
    for index, paragraphs in enumerate(pages):
        height = document[index].rect.height
        keys.update({_running_key(p) for p in paragraphs if _in_margin(p, height)})
    threshold = max(3, int(len(pages) * 0.3))
    running = {key for key, count in keys.items() if count >= threshold}

    sections = _sections(document.get_toc(simple=True), len(pages))
    result = []
    for index, paragraphs in enumerate(pages):
        page = document[index]
        width, height = page.rect.width, page.rect.height
        for paragraph in paragraphs:
            if _in_margin(paragraph, height) and _running_key(paragraph) in running:
                continue
            if not is_valid(paragraph.text):
                continue
            result.append(
                {
                    "page": paragraph.page,
                    "seq": len(result),
                    "section": sections[index],
                    "bbox": _fractions(paragraph.rect, width, height),
                    "text": paragraph.text,
                    "sentences": [
                        {
                            "seq": number,
                            "start": start,
                            "end": end,
                            "page": paragraph.page,
                            "quads": _quads(paragraph, start, end, width, height),
                        }
                        for number, (start, end) in enumerate(sentences(paragraph.text))
                    ],
                }
            )
    return result


def embedding_text(paragraph: dict) -> str:
    """The paragraph with its section title prefixed, as monad-knowledge's chunker embeds it."""
    return f"{paragraph['section']}\n{paragraph['text']}" if paragraph.get("section") else paragraph["text"]


def links(embeddings, k: int, floor: float) -> List[dict]:
    """`next` in reading order, and `similar` to the k nearest other paragraphs above `floor`."""
    import numpy as np

    count = len(embeddings)
    result = [{"source": index, "target": index + 1, "kind": "next"} for index in range(count - 1)]
    if count < 2 or k <= 0:
        return result

    vectors = np.asarray(embeddings, dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    for first in range(0, count, 512):
        scores = vectors[first : first + 512] @ vectors.T
        for row, source in enumerate(range(first, min(first + 512, count))):
            scores[row, source] = -1.0
            for target in np.argsort(-scores[row])[:k]:
                score = float(scores[row, target])
                if score < floor:
                    break
                result.append({"source": source, "target": int(target), "kind": "similar", "score": round(score, 4)})
    return result
