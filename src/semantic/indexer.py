"""
Main semantic indexer - orchestrates embedding generation and storage
"""
from typing import List, Tuple, Dict
import logging

from semantic.embeddings.embedding_generator import EmbeddingGenerator
from semantic.storage.milvus_manager import MilvusManager
from config.semantic_config import SemanticConfig

logger = logging.getLogger(__name__)


class SemanticIndexer:
    """
    Orchestrates the semantic indexing pipeline:
    1. Takes extracted text from TextHandler
    2. Generates embeddings
    3. Stores in Milvus
    """
    
    def __init__(self):
        """Initialize indexer with embedding generator and Milvus manager"""
        self.embedding_generator = EmbeddingGenerator()
        self.milvus_manager = MilvusManager()
        logger.info("✓ SemanticIndexer initialized")
    
    def index_document(
        self,
        document_id: str,
        pages: List[Dict]
    ) -> Dict:
        """
        Index a document from TextHandler output.

        Args:
            document_id: Unique document identifier
            pages: List of page dicts from TextHandler with structure:
                {
                    'page_num': int,
                    'text': str,
                    'paragraphs': List[str],
                    'sentences': List[str]
                }

        Returns:
            Dict with indexing results
        """
        logger.info(f"Indexing document: {document_id}")

        # Prepare chunks and metadata
        chunks, metadata = self._prepare_chunks(pages)
        
        if not chunks:
            logger.warning(f"No valid chunks found for document {document_id}")
            return {
                "document_id": document_id,
                "success": False,
                "chunks_indexed": 0,
                "error": "No valid chunks"
            }
        
        logger.info(f"Prepared {len(chunks)} chunks")
        
        # Generate embeddings
        try:
            embeddings = self.embedding_generator.generate_embeddings(
                chunks,
                normalize=True,
                show_progress=True
            )
            logger.info(f"✓ Generated {len(embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return {
                "document_id": document_id,
                "success": False,
                "chunks_indexed": 0,
                "error": str(e)
            }
        
        # Store in Milvus
        try:
            inserted_ids = self.milvus_manager.insert_embeddings(
                document_id=document_id,
                embeddings=embeddings,
                metadata=metadata
            )
            
            logger.info(f"✓ Document {document_id} indexed successfully")
            
            return {
                "document_id": document_id,
                "success": True,
                "chunks_indexed": len(inserted_ids),
                "chunk_level": SemanticConfig.CHUNK_LEVEL,
                "embedding_dim": SemanticConfig.EMBEDDING_DIM
            }
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            return {
                "document_id": document_id,
                "success": False,
                "chunks_indexed": 0,
                "error": str(e)
            }
    
    def _prepare_chunks(
        self,
        pages: List[Dict]
    ) -> Tuple[List[str], List[Dict]]:
        """
        Prepare chunks and metadata from TextHandler output.

        Args:
            pages: List of page dicts with 'page_num', 'text', 'paragraphs', 'sentences'

        Returns:
            (chunks, metadata) tuple
        """
        chunks = []
        metadata = []

        if SemanticConfig.CHUNK_LEVEL == "paragraph":
            # Paragraph-level chunking
            for page_dict in pages:
                page_num = page_dict['page_num']
                page_paragraphs = page_dict['paragraphs']

                for para_idx, para_text in enumerate(page_paragraphs):
                    # Filter valid paragraphs
                    if self._is_valid_paragraph(para_text):
                        chunks.append(para_text)
                        metadata.append({
                            "page_num": page_num,
                            "paragraph_idx": para_idx,
                            "chunk_type": "paragraph",
                            "text": para_text,
                            "text_length": len(para_text)
                        })

        elif SemanticConfig.CHUNK_LEVEL == "page":
            # Page-level chunking
            for page_dict in pages:
                page_num = page_dict['page_num']
                page_text = page_dict['text']

                if page_text and page_text.strip():
                    chunks.append(page_text)
                    metadata.append({
                        "page_num": page_num,
                        "paragraph_idx": -1,
                        "chunk_type": "page",
                        "text": page_text[:2000],  # Preview only
                        "text_length": len(page_text)
                    })

        logger.debug(f"Prepared {len(chunks)} chunks at {SemanticConfig.CHUNK_LEVEL} level")
        return chunks, metadata
    
    def _is_valid_paragraph(self, paragraph: str) -> bool:
        """Check if paragraph should be indexed"""
        if not paragraph or not paragraph.strip():
            return False
        
        # Minimum length check (words)
        word_count = len(paragraph.split())
        if word_count < SemanticConfig.MIN_PARAGRAPH_LENGTH:
            return False
        
        # Filter out common boilerplate
        lower_para = paragraph.lower()
        boilerplate_patterns = [
            "table of contents",
            "obsah",  # Slovak for contents
            "zoznam obrázkov",  # List of figures
            "zoznam tabuliek",  # List of tables
        ]
        
        if any(pattern in lower_para for pattern in boilerplate_patterns):
            return False
        
        return True
    
    def delete_document(self, document_id: str) -> Dict:
        """Delete document from index"""
        try:
            deleted_count = self.milvus_manager.delete_by_document_id(document_id)
            
            return {
                "document_id": document_id,
                "success": True,
                "chunks_deleted": deleted_count
            }
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return {
                "document_id": document_id,
                "success": False,
                "error": str(e)
            }