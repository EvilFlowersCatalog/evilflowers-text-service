import os
import pandas as pd

from domain.base.ExtractorInterface import ExtractorInterface
from domain.extractors.PdfPlumberExtractor import PdfPlumberExtractor
# from domain.extractors.CamelotExtractor import CamelotExtractor
from config.Config import Config

class TableExtractor:

    config = Config()
    _document_path: str
    _extractor: ExtractorInterface
    tables: dict[str, pd.DataFrame]
    dataframe_mode: bool

    def __init__(self, document_path: str, csv_mode: bool = False):
        self._validate(document_path)
        self._document_path = document_path
        self._extractor = self._load_extractor()
        self.dataframe_mode = True
        self.csv_mode = csv_mode

    def set_document_path(self, document_path: str):
        self._document_path = document_path
    
    def extract_tables(self):
        self.tables = self._extractor.extract_tables()
        return self.tables

    # Private methods
    def _validate(self, document_path: str):
        assert os.path.exists(
            document_path
        ), f"Document path did not found: {document_path}"

    def _load_extractor(self):
        return PdfPlumberExtractor(self._document_path)
