from typing import Optional
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from langdetect import detect, LangDetectException
from config.Config import Config


class TextProcessor:
    """
    Processes extracted text into embedding-ready chunks.
    
    - Language detection (Slovak vs English)
    - Semantic chunking with sentence awareness
    - Chunk overlap for context preservation
    - Metadata attachment per chunk
    """
  
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n"
        )

    def process_text(
        self,
        pages: list[dict],
        toc: Optional[list] = None,
        doc_id: str = "unknown"
    ) -> dict:
        """
        Main entry point - processes TextExtractor output into chunks.
        Pages format: [{'page_num': 1, 'text': '...', 'paragraphs': [...], 'sentences': [...]}, ...]
        """
        # Extract text from new dict format
        full_text = " ".join([p['text'] for p in pages if p.get('text')])
        
        if not full_text.strip():
            return {
                'doc_id': doc_id,
                'language': 'unknown',
                'chunks': [],
                'total_chunks': 0
            }
        
        language = self._detect_language(full_text)
        page_offsets = self._build_page_offsets(pages)
        
        # Create LlamaIndex document and split
        document = Document(text=full_text)
        nodes = self.splitter.get_nodes_from_documents([document])
        
        chunks = []
        for i, node in enumerate(nodes):
            source_page = self._find_source_page(node.start_char_idx or 0, page_offsets)
            section = self._find_section(source_page, toc) if toc else None
            
            chunks.append({
                'text': node.text,
                'metadata': {
                    'doc_id': doc_id,
                    'language': language,
                    'chunk_index': i,
                    'source_page': source_page,
                    'section': section or "",  
                    'word_count': len(node.text.split()),
                    'text': node.text 
                }
            })
        
        return {
            'doc_id': doc_id,
            'language': language,
            'chunks': chunks,
            'total_chunks': len(chunks)
        }

    def _build_page_offsets(self, pages: list) -> list:
        """Build map of (start_offset, end_offset, page_num) from dict format"""
        offsets = []
        current_offset = 0
        
        for page in pages:
            page_text = page.get('text', '')
            page_num = page.get('page_num', 1)
            
            if page_text:
                text_len = len(page_text) + 1  # +1 for the space we added in join
                offsets.append((current_offset, current_offset + text_len, page_num))
                current_offset += text_len
        
        return offsets

    def _detect_language(self, text: str) -> str:
        """Detect primary language. Returns 'sk', 'en', or detected code."""
        sample_size = 1000
        if len(text) > sample_size * 2:
            sample = text[:sample_size] + " " + text[len(text)//2:len(text)//2 + sample_size]
        else:
            sample = text[:sample_size]
        
        try:
            return detect(sample)
        except LangDetectException:
            return "unknown"

    def _find_source_page(self, char_offset: int, page_offsets: list) -> int:
        """Find which page a character offset belongs to"""
        for start, end, page_num in page_offsets:
            if start <= char_offset < end:
                return page_num
        return page_offsets[0][2] if page_offsets else 1

    def _find_section(self, page_num: int, toc: list) -> Optional[str]:
        """Find TOC section for a page. TOC format: [[level, title, page], ...]"""
        if not toc:
            return None
            
        current_section = None
        for entry in toc:
            if len(entry) >= 3 and entry[2] <= page_num:
                current_section = entry[1]
            elif len(entry) >= 3:
                break
        
        return current_section