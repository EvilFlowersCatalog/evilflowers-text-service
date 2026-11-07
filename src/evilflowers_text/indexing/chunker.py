"""Page text → embedding-ready chunks for search-service (language, sentence-aware chunks, metadata)."""

from typing import Any

from evilflowers_text.indexing.text_extractor import Page, TocEntry

#: Characters sampled (from the start and the middle) to detect the language.
LANGUAGE_SAMPLE = 1000


def detect_language(text: str) -> str:
    """ISO code of the primary language, or ``unknown``."""
    from langdetect import LangDetectException, detect

    if len(text) > LANGUAGE_SAMPLE * 2:
        middle = len(text) // 2
        sample = text[:LANGUAGE_SAMPLE] + " " + text[middle : middle + LANGUAGE_SAMPLE]
    else:
        sample = text[:LANGUAGE_SAMPLE]
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


def page_offsets(pages: list[Page]) -> list[tuple[int, int, int]]:
    """``(start, end, page_num)`` of each page's text in the joined document text."""
    offsets = []
    position = 0
    for page in pages:
        if text := page.get("text", ""):
            length = len(text) + 1  # the space of the join
            offsets.append((position, position + length, page.get("page_num", 1)))
            position += length
    return offsets


def source_page(offset: int, offsets: list[tuple[int, int, int]]) -> int:
    for start, end, page in offsets:
        if start <= offset < end:
            return page
    return offsets[0][2] if offsets else 1


def section(page: int, toc: list[TocEntry] | None) -> str | None:
    """The last ToC heading starting on or before `page` (ToC entries are ``[level, title, page]``)."""
    current = None
    for entry in toc or []:
        if len(entry) < 3:
            continue
        if entry[2] > page:
            break
        current = entry[1]
    return current


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        from llama_index.core.node_parser import SentenceSplitter

        self.splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" ", paragraph_separator="\n\n"
        )

    def chunk(self, pages: list[Page], toc: list[TocEntry] | None = None, doc_id: str = "unknown") -> dict[str, Any]:
        """The search-service `/index` payload's `chunks`."""
        from llama_index.core import Document

        full_text = " ".join(page["text"] for page in pages if page.get("text"))
        if not full_text.strip():
            return {"doc_id": doc_id, "language": "unknown", "chunks": [], "total_chunks": 0}

        language = detect_language(full_text)
        offsets = page_offsets(pages)
        chunks = []
        for index, node in enumerate(self.splitter.get_nodes_from_documents([Document(text=full_text)])):
            page = source_page(node.start_char_idx or 0, offsets)
            chunks.append(
                {
                    "text": node.text,
                    "metadata": {
                        "doc_id": doc_id,
                        "language": language,
                        "chunk_index": index,
                        "source_page": page,
                        "section": section(page, toc) or "",
                        "word_count": len(node.text.split()),
                        "text": node.text,
                    },
                }
            )
        return {"doc_id": doc_id, "language": language, "chunks": chunks, "total_chunks": len(chunks)}
