import os
import fitz
import ocrmypdf
import tempfile
import re
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from config.Config import Config
from io import BytesIO

# EDITOR -> we need to consider how this would work on the server + what model to use, test them
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt_tab/slovak.pickle')
except LookupError:
    nltk.download('punkt_tab')

class TextExtractor:
    config = Config()
    _document_path: str

    def __init__(self, document_path: str):
        self._validate(document_path)
        self._document_path = document_path

    # EDITOR -> Wtf is this for
    def set_document_path(self, document_path: str):
        self._document_path = document_path

    ##
    # Fitz/Digital text based functions
    def extract(self) -> tuple[list[dict[str, any]], list | None]:
        """
        Extract everything in one pass through the PDF
        Returns: (pages, paragraphs, metadata)
        """
        # EDITOR -> maybe add json structure instead of list index alignment its more human readable
        # {
        #     'page_num': 1,
        #     'text': "full page text...",
        #     'paragraphs': [
        #         {
        #             'text': "paragraph text...",
        #             'sentences': ["sent1", "sent2"]
        #         }
        #     ]
        # }
        doc = self._load_document()

        if self._is_digital(doc):
            pages, toc = self._extract_digital(doc)
        else:
            pages, toc = self._extract_ocr()
        doc.close()
    
        return pages,  toc

    ##
    # Private functions
    # EDITOR -> Add python esque looping like in the extract digital function
    def _is_digital(self, doc: fitz.Document) -> bool:
        """Check if PDF has extractable text or needs OCR"""
        # Sample first few pages
        for page_num in range(min(30, doc.page_count)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            if len(text) > 100:  # arbitrary threshold
                return True
        return False

    def _check_toc(self, pages: list[dict[str, any]]) -> bool:  
        toc_keywords = ["Table of Contents", "Chapter", "Section", "Contents"]
        for page in pages:
            text = page[1]
            if text:
                for keyword in toc_keywords:
                    if keyword in text:
                        return True
        return False
    
    def _extract_toc(self, pages: list[dict[str, any]]) -> any:
        # TODO: Implement TOC extraction
        pass

    def _load_document(self) -> fitz.Document:
        doc = fitz.open(self._document_path)
        return doc
    
    def _validate(self, document_path: str):
        assert os.path.exists(
            document_path
        ), f"Document path did not found: {document_path}"

    def _clean_whitespace(self, text: str) -> str:
        # Fix hyphenated line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        text = re.sub(r'\n', ' ', text)  # Replace remaining newlines with spaces
        text = re.sub(r' +', ' ', text)  # Collapse multiple spaces
        
        return text.strip()
    
    def _extract_digital(self, doc: fitz.Document) -> tuple[list[dict[str, any]], list]:
        toc = doc.get_toc()

        pages = []
        # paragraphs = []
        # sentences = []
        
        for page_num, page in enumerate(doc, start=1):
            # Get blocks (includes position + text)
            blocks = page.get_text("blocks")
            
            # Extract page-level text (concatenated blocks)
            cleaned_blocks = [
                self._clean_whitespace(block[4]) 
                for block in blocks
            ]

            page_text = " ".join(cleaned_blocks)
            # pages.append([page_num, page_text])
            
            # Extract paragraph-level text (individual blocks)
            page_paragraphs = [t for t in cleaned_blocks if len(t) > 20]
            # paragraphs.append(page_paragraphs)

            # Extract page-level sentences using NLP
            page_sentences = sent_tokenize(page_text)
            # sentences.append(page_sentences)

            pages.append({
                'page_num': page_num,
                'text': page_text,
                'paragraphs': page_paragraphs,
                'sentences': page_sentences
            })

        return pages, toc

    def extract_toc(self, pages: list[dict[str, any]], found_toc: bool) -> any:
        if found_toc:
            return self._extract_toc(pages)
        # TODO: This will find the TOC but when it will not be in metadata it will be complicated to extract
        # if self._check_toc(pages):
        #     return self._extract_toc(pages)
        else:
            return None
        
    ## OCR based functions
    def _extract_ocr(self) -> tuple[list[dict[str, any]], None]:
        pages = []
        # paragraphs = []
        # sentences = []

        # try:
        # Process PDF with OCR and get bytes directly
        ocr_pdf_bytes = self._process_pdf_with_ocr(self._document_path)

        # Convert PDF bytes to images directly
        images = convert_from_bytes(ocr_pdf_bytes)
        
        for page_num, img in enumerate(images, 1):
            # Extract text from the current page
            page_text = pytesseract.image_to_string(img)
            # pages.append([page_num, page_text])

            # Split into paragraphs
            page_paragraphs = self._split_into_paragraphs(page_text)            
            # paragraphs.append(page_paragraphs)

            page_sentences = []
            for paragraph_num, paragraph in enumerate(page_paragraphs):
                paragraph_text = paragraph
                paragraph_sentences = paragraph_text.split(".")
                paragraph_sentence = []
                for i, sentence in enumerate(paragraph_sentences):
                    if sentence.strip():  # Only add non-empty sentences
                        paragraph_sentence.append(sentence.strip())
                page_sentences.append(paragraph_sentence)
            # sentences.append(page_sentences)

            pages.append({
                'page_num': page_num,
                'text': page_text,
                'paragraphs': page_paragraphs,
                'sentences': page_sentences
            })

        # except Exception as e:
        return pages, None

    def _process_pdf_with_ocr(self, pdf_path: str) -> bytes:
        """Process PDF with OCR and return the bytes directly"""
        output_buffer = BytesIO()
        
        ocrmypdf.ocr(pdf_path, output_buffer,
                    force_ocr=True, 
                    skip_text=False,
                    deskew=True,
                    clean=True)
        
        return output_buffer.getvalue()
    
    def _split_into_paragraphs(self, text: str) -> list[str]:
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
