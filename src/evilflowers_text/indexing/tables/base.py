"""Whole-document table extraction of the indexing pipeline (not the IP-018 `tables` job)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class TableExtractor(ABC):
    """Every table of a PDF as a data frame, keyed ``<file stem>_page_<nnn>_table_<nn>``."""

    def __init__(self, mode: str | None = None):
        self.mode = mode

    @abstractmethod
    def extract(self, path: Path) -> dict[str, "pd.DataFrame"]: ...
