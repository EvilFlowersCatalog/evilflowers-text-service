from text_handler.TextExtractor import TextExtractor
from text_handler.TextProcessor import TextProcessor
from text_handler.TableExtractor import TableExtractor

class TextHandler:

    _instance = None
    table_extractor: TableExtractor
    text_extractor: TextExtractor
    text_processor: TextProcessor
    table_extract_csv: bool

    def __init__(self, document_path: str, table_extract_csv: bool = False):
        if not hasattr(self, "initialized"):
            self.table_extract_csv = table_extract_csv
            self.text_extractor = TextExtractor(document_path)
            self.table_extractor = TableExtractor(document_path, self.table_extract_csv)
            self.text_processor = TextProcessor()
            self.initialized = True
        else:
            return self._instance

    def extract_tables(self):
        tables = self.table_extractor.extract_tables()
        return tables
    
    def extract_text(self, found_toc: bool):
        pages, toc = self.text_extractor.extract()

        return pages, toc

    def process_text(self, text: str):
        return self.text_processor.process_text(text)
