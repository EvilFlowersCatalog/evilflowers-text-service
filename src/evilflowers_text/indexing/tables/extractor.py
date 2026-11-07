"""The table extractor named by `TABLE_EXTRACTOR`; a new one is a new entry here."""

from evilflowers_text.indexing.tables.base import TableExtractor
from evilflowers_text.indexing.tables.camelot import CamelotExtractor
from evilflowers_text.indexing.tables.pdfplumber import PdfPlumberExtractor

EXTRACTORS: dict[str, type[TableExtractor]] = {
    "PdfPlumber": PdfPlumberExtractor,
    "Camelot": CamelotExtractor,
}


def table_extractor(name: str, mode: str | None = None) -> TableExtractor:
    try:
        return EXTRACTORS[name](mode)
    except KeyError:
        raise ValueError(f"Unknown table extractor {name!r}; expected one of {sorted(EXTRACTORS)}") from None
