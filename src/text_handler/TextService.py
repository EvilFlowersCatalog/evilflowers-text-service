from text_handler.TextExtractor import TextExtractor
from text_handler.TextProcessor import TextProcessor
from text_handler.TableExtractor import TableExtractor

class TextHandler:

    _instance = None
    table_extractor: TableExtractor
    text_extractor: TextExtractor
    text_processor: TextProcessor

    def __init__(self, document_path: str):
        if not hasattr(self, "initialized"):
            self.text_extractor = TextExtractor(document_path)
            self.table_extractor = TableExtractor()
            self.text_processor = TextProcessor()
            self.initialized = True
        else:
            return self._instance

    def extract_tables(self):
        tables = self.table_extractor.extract_tables()
        return tables
    
    def extract_text(self):
        pages = self.text_extractor.extract_pages_text()
        paragraphs = self.text_extractor.extract_paragraphs()
        sentences = self.text_extractor.extract_sentences(paragraphs)
        return pages, paragraphs, sentences

    def process_text(self, text: str):
        return self.text_processor.process_text(text)
