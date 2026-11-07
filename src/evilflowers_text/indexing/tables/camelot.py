from pathlib import Path

from evilflowers_text.indexing.tables.base import TableExtractor


class CamelotExtractor(TableExtractor):
    """Camelot's `lattice` (ruled tables) or `stream` flavour; needs `camelot-py`, not in the image."""

    def extract(self, path: Path) -> dict:
        import camelot

        tables = camelot.read_pdf(str(path), pages="all", flavor=self.mode or "lattice")
        return {
            f"{path.stem}_page_{int(table.page):03d}_table_{index:02d}": table.df
            for index, table in enumerate(tables, start=1)
        }
