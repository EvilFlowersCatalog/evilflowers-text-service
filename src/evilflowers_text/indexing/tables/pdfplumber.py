from pathlib import Path

import pdfplumber

from evilflowers_text.indexing.tables.base import TableExtractor


class PdfPlumberExtractor(TableExtractor):
    def extract(self, path: Path) -> dict:
        import pandas as pd

        frames = {}
        with pdfplumber.open(path) as document:
            for page_number, page in enumerate(document.pages, start=1):
                for index, table in enumerate(page.extract_tables(), start=1):
                    if table:
                        frames[f"{path.stem}_page_{page_number:03d}_table_{index:02d}"] = pd.DataFrame(
                            table[1:], columns=table[0]
                        )
        return frames
