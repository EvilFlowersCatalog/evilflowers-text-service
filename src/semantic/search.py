"""
Semantic search interface
"""
from typing import List, Dict, Optional
import logging

from semantic.embeddings.embedding_generator import EmbeddingGenerator
from semantic.storage.milvus_manager import MilvusManager

logger = logging.getLogger(__name__)


class SemanticSearch:
    """Search interface for semantic queries"""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.milvus_manager = MilvusManager()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: Optional[str] = None,
        page_num: Optional[int] = None
    ) -> List[Dict]:
        """
        Semantic search across indexed documents.
        
        Args:
            query: Search query in any supported language
            top_k: Number of results
            document_id: Filter by document (optional)
            page_num: Filter by page (optional)
            
        Returns:
            List of search results
        """
        logger.info(f"Searching: '{query[:50]}...'")
        
        # Generate query embedding
        try:
            query_embedding = self.embedding_generator.generate_single_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []
        
        # Search in Milvus
        try:
            results = self.milvus_manager.search(
                query_embedding=query_embedding,
                top_k=top_k,
                document_id=document_id,
                page_num=page_num
            )
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []