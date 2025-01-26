import os
import pandas as pd

from abc import ABC, abstractmethod

class ExtractorInterface(ABC):
    def __init__(self, document_path: str, mode: str = None):
        self._validate(document_path)
        self._document_path = document_path
        self._mode = mode

    @abstractmethod
    def extract_tables(self, dataframe: bool = True, csv: bool = False, output_path: str = ".") -> dict[str, pd.DataFrame]:
        pass

    # Private methods
    def _validate(self, document_path: str):
        assert os.path.exists(
            document_path
        ), f"Document path not found: {document_path}"

    @abstractmethod
    def _load_document(self):
        pass