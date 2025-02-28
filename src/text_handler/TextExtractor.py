import os
import fitz

import ocrmypdf
import tempfile
import os
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes

from config.Config import Config
from io import BytesIO

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
    
    def extract_pages_text(self):
        pages = self.extract_pages()
        pages_text = []
        for page in pages:
            page_num = page[0]
            page_obj = page[1]  # Get the actual page object
            page_text = page_obj.get_text("text")  # Extract text from page object
            pages_text.append([page_num, page_text])
        return pages_text

    def extract_paragraphs(self):
        """Process PDF and extract paragraphs page by page"""
        try:
            # Process PDF with OCR and get bytes directly
            ocr_pdf_bytes = self._process_pdf_with_ocr(self._document_path)
            
            # Convert PDF bytes to images directly
            images = convert_from_bytes(ocr_pdf_bytes)
            # Dictionary to store paragraphs by page
            all_pages_paragraphs = []
            
            # Process each page
            for page_num, img in enumerate(images, 1):
                # Extract text from the current page
                page_text = pytesseract.image_to_string(img)
                
                # Split into paragraphs
                paragraphs = self._split_into_paragraphs(page_text)
                
                # Store paragraphs with page number
                all_pages_paragraphs.append(paragraphs)

            return all_pages_paragraphs
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    
    def extract_sentences(self, all_paragraphs):
        print("all paragraphs: ", all_paragraphs)
        paragraphs = all_paragraphs
        sentences = []
        for page_num, page in enumerate(paragraphs):
            page_sentences = []
            for paragraph_num, paragraph in enumerate(page):
                paragraph_text = paragraph
                paragraph_sentences = paragraph_text.split(".")
                paragraph_sentence = []
                for i, sentence in enumerate(paragraph_sentences):
                    if sentence.strip():  # Only add non-empty sentences
                        paragraph_sentence.append(sentence.strip())
                page_sentences.append(paragraph_sentence)
            sentences.append(page_sentences)

        return sentences
    
    def extract_toc(self, pages, found_toc: bool):
        if found_toc:
            return self._extract_toc(pages)
        # TODO: This will find the TOC but when it will not be in metadata it will be complicated to extract
        # if self._check_toc(pages):
        #     return self._extract_toc(pages)
        else:
            return None

    ##
    # Private functions
    def _check_toc(self, pages):
        toc_keywords = ["Table of Contents", "Chapter", "Section", "Contents"]
        for page in pages:
            text = page[1]
            if text:
                for keyword in toc_keywords:
                    if keyword in text:
                        return True
        return False
    
    def _extract_toc(self, pages):
        # TODO: Implement TOC extraction
        pass

    def _process_pdf_with_ocr(self, pdf_path):
        """Process PDF with OCR and return the bytes directly"""
        output_buffer = BytesIO()
        
        ocrmypdf.ocr(pdf_path, output_buffer,
                    force_ocr=True, 
                    skip_text=False,
                    deskew=True,
                    clean=True)
        
        return output_buffer.getvalue()

    def _load_document(self):
        doc = fitz.open(self._document_path)
        return doc
    
    def _validate(self, document_path: str):
        assert os.path.exists(
            document_path
        ), f"Document path did not found: {document_path}"

    def _split_into_paragraphs(self, text):
        """ Split text into paragraphs with improved handling """
        raw_paragraphs = text.split('\n\n')
        
        paragraphs = []
        current_paragraph = []
        
        # split text into paragraphs
        for para in raw_paragraphs:
            if not para.strip():
                continue
                
            # clean text
            cleaned = ' '.join(para.split())
            
            # if it's a header or very short line, keep it separate
            if len(cleaned) < 50 and (
                any(word in cleaned.lower() for word in ['abstract', 'introduction', 'conclusion', 'references']) or
                cleaned.isupper() or
                any(word in cleaned.lower() for word in ['department', 'school', 'email', 'university'])
            ):
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                paragraphs.append(cleaned)
            else:
                # if it looks like a continuation of a sentence, append to current paragraph
                if current_paragraph and not cleaned[0].isupper() and len(current_paragraph[-1]) > 0 and current_paragraph[-1][-1] not in '.!?':
                    current_paragraph.append(cleaned)
                else:
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                    current_paragraph.append(cleaned)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return [p for p in paragraphs if p.strip()]  # remove empty paragraphs

