import os
import re
import numpy as np
import fitz
import ocrmypdf
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from doclayout_yolo import YOLOv10
from config.Config import Config
from io import BytesIO

class TextExtractor:
    config = Config()
    _document_path: str

    def __init__(self, document_path: str):
        self._document_path = document_path

    # Public functions
    def extract(self) -> tuple[list[dict[str, any]], list | None]:
        """
        Extract everything in one pass through the PDF
        Returns: (pages, paragraphs, metadata)
        """
        doc = fitz.open(self._document_path)

        if self._is_digital(doc):
            pages, toc = self._extract_digital(doc)
        else:
            pages, toc = self._extract_ocr()
        doc.close()
    
        return pages,  toc

    # Private functions
    # EDITOR -> Add python esque looping like in the extract digital function
    def _is_digital(self, doc: fitz.Document) -> bool:
        import random
        total = doc.page_count
        start = int(total * 0.1)
        end = int(total * 0.9)
        population = range(start, end)
        sample_size = min(20, len(population))
        pages = random.Random(42).sample(list(population), sample_size)
        digital_pages = 0
        for page_num in pages:
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            if len(text) > 100:
                digital_pages += 1
        return (digital_pages / sample_size) >= 0.5
    
    def _extract_toc(self, source: fitz.Document | list) -> list | None:
        if isinstance(source, fitz.Document):
            toc = source.get_toc()
            if toc is not None:
                return toc

        yolo = YOLOv10(Config.YOLO_MODEL_PATH)

        if isinstance(source, fitz.Document):
            pages_to_scan = range(min(20, source.page_count))

            for page_idx in pages_to_scan:
                page = source.load_page(page_idx)
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                list_items = self._get_list_items(yolo, img)
                if not list_items:
                    continue

                blocks = page.get_text("blocks")
                toc = []
                for x1, y1, x2, y2 in list_items:
                    margin = 5
                    node_text = " ".join(
                        self._clean_whitespace(b[4]) for b in blocks
                        if x1 - margin <= b[0] <= x2 + margin and y1 - margin <= b[1] <= y2 + margin
                    ).strip()
                    entry = self._parse_toc_line(node_text, x1)
                    if entry:
                        toc.append(entry)
                return toc if toc else None

        else:
            for img in source[:10]:
                list_items = self._get_list_items(yolo, img)
                if not list_items:
                    continue

                toc = []
                for x1, y1, x2, y2 in list_items:
                    cropped = img.crop((x1, y1, x2, y2))
                    node_text = pytesseract.image_to_string(cropped).strip()
                    entry = self._parse_toc_line(node_text, x1)
                    if entry:
                        toc.append(entry)
                return toc if toc else None

        return None

    def _get_list_items(self, yolo, img) -> list:
        results = yolo.predict(source=img, imgsz=1120, conf=0.2, iou=0.5, agnostic_nms=True, verbose=False)
        result = results[0]
        return sorted(
            [box.xyxy[0].tolist() for box in result.boxes if result.names[int(box.cls[0].item())] == 'List-item'],
            key=lambda b: b[1]
        )

    def _parse_toc_line(self, text: str, x1: float) -> list | None:
        if not text:
            return None
        match = re.search(r'^(.*?)\s*[\.\s]{2,}\s*(\d+)\s*$', text) or \
                re.search(r'^(.*?)\s+(\d+)\s*$', text)
        if not match:
            return None
        return [1 + int(x1 // 50), match.group(1).strip(), int(match.group(2))]
        
    def _clean_whitespace(self, text: str) -> str:
        # Fix hyphenated line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        text = re.sub(r'\n', ' ', text)  # Replace remaining newlines with spaces
        text = re.sub(r' +', ' ', text)  # Collapse multiple spaces
        
        return text.strip()
    
    def _extract_digital(self, doc: fitz.Document) -> tuple[list[dict[str, any]], list]:
        toc = self._extract_toc(doc)

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
            # page_sentences = sent_tokenize(page_text)
            # sentences.append(page_sentences)

            pages.append({
                'page_num': page_num,
                'text': page_text,
                'paragraphs': page_paragraphs,
            })

        return pages, toc
        
    ## OCR based functions
    def _extract_ocr(self) -> tuple[list[dict[str, any]], None]:
        ocr_pdf_bytes = self._process_pdf_with_ocr(self._document_path)
        doc = fitz.open(stream=ocr_pdf_bytes, filetype="pdf")
        
        images = [
            Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            for page in doc
            for pix in [page.get_pixmap(dpi=150)]
        ]
        toc = self._extract_toc(images)
        
        pages = []
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            cleaned_blocks = [self._clean_whitespace(b[4]) for b in blocks]
            page_text = " ".join(cleaned_blocks)
            page_paragraphs = [t for t in cleaned_blocks if len(t) > 20]
            pages.append({
                'page_num': page_num,
                'text': page_text,
                'paragraphs': page_paragraphs,
            })
        doc.close()
        
        return pages, toc

    def _process_pdf_with_ocr(self, pdf_path: str) -> bytes:
        """Process PDF with OCR and return the bytes directly"""
        output_buffer = BytesIO()
        
        ocrmypdf.ocr(pdf_path, output_buffer,
                    force_ocr=True,
                    deskew=True,
                    optimize=0,
                    output_type='pdf',        # skip PDF/A generation
                    fast_web_view=999999)     # skip fast web view optimization
        
        return output_buffer.getvalue()
    
    ## USELESS
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
