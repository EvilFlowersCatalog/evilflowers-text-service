import os
import fitz

from config.Config import Config

class TextExtractor:

    config = Config()
    _document_path: str

    def __init__(self, document_path: str):
        self._validate(document_path)
        self._document_path = document_path

    def set_document_path(self, document_path: str):
        self._document_path = document_path

    def extract_pages(self):
        doc = self._load_document()

        pages = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pages.append([page_num, page])

        return pages

    def extract_paragraphs(self):
        doc = self._load_document()

        all_paragraphs = []

        # Iterate over pages
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            # Extract text as lines
            text = page.get_text("text")
            lines = text.split("\n")
            
            # Group lines into paragraphs
            paragraphs = []
            paragraph = []
            
            for line in lines:
                if line.strip():  # If the line is not empty
                    paragraph.append(line)
                else:
                    # If we encounter an empty line, consider the paragraph complete
                    if paragraph:
                        paragraphs.append([page_num, " ".join(paragraph)])
                        paragraph = []
            
            # Add the last paragraph if it exists
            if paragraph:
                paragraphs.append([page_num, " ".join(paragraph)])
            
            # Append to all paragraphs
            all_paragraphs.extend(paragraphs)

        return all_paragraphs
    
    def extract_sentences(self):
        paragraphs = self.extract_paragraphs()
        sentences = []
        for paragraph_num, paragraph in enumerate(paragraphs):
            page_num = paragraph[0]
            paragraph_text = paragraph[1]
            paragraph_sentences = paragraph_text.split(".")
            for i, sentence in enumerate(paragraph_sentences):
                if sentence.strip():  # Only add non-empty sentences
                    sentences.append([page_num, paragraph_num, sentence.strip()])
        return sentences

    ##
    # Private functions
    def _load_document(self):
        doc = fitz.open(self._document_path)
        return doc
    
    def _validate(self, document_path: str):
        assert os.path.exists(
            document_path
        ), f"Document path did not found: {document_path}"

