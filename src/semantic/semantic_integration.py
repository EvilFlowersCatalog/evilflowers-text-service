"""
Enhanced TextService Integration with Semantic Search
Extends the existing TextHandler to include semantic embedding generation and vector storage
"""

import os
import logging
from typing import Optional, Dict, List, Tuple

from semantic_embeddings import get_embeddings_service
from milvus_client import create_milvus_client
from semantic_search_service import SemanticSearchService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticTextHandler:
    """
    Extended TextHandler with semantic search capabilities.
    Integrates with existing text extraction pipeline to add vector embeddings.
    """
    
    def __init__(
        self,
        text_handler,  # Your existing TextHandler instance
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        enable_semantic_search: bool = True
    ):
        """
        Initialize semantic text handler.
        
        Args:
            text_handler: Existing TextHandler instance
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            enable_semantic_search: Enable semantic search features
        """
        self.text_handler = text_handler
        self.enable_semantic_search = enable_semantic_search
        
        if self.enable_semantic_search:
            self.semantic_service = SemanticSearchService(
                milvus_host=milvus_host,
                milvus_port=milvus_port
            )
            logger.info("Semantic search enabled")
        else:
            self.semantic_service = None
            logger.info("Semantic search disabled")
    
    def extract_and_index(
        self,
        document_id: str,
        found_toc: bool = False,
        chunk_level: str = "paragraph"
    ) -> Dict:
        """
        Extract text and generate semantic embeddings in one pass.
        
        Args:
            document_id: Unique document identifier
            found_toc: Whether TOC was found
            chunk_level: Embedding granularity ("page" or "paragraph")
            
        Returns:
            Dictionary with extraction and indexing results
        """
        # Extract text using existing service
        pages, paragraphs, sentences, toc = self.text_handler.extract_text(found_toc)
        
        result = {
            "document_id": document_id,
            "extraction": {
                "pages_extracted": len(pages),
                "paragraphs_extracted": sum(len(p) for p in paragraphs),
                "toc_found": toc is not None
            }
        }
        
        # Generate and store embeddings if enabled
        if self.enable_semantic_search and self.semantic_service:
            indexing_result = self.semantic_service.index_document(
                document_id=document_id,
                pages=pages,
                paragraphs=paragraphs,
                chunk_level=chunk_level
            )
            result["indexing"] = indexing_result
        
        return result
    
    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        Perform semantic search.
        
        Args:
            query_text: Search query
            top_k: Number of results
            
        Returns:
            Search results
        """
        if not self.enable_semantic_search:
            raise RuntimeError("Semantic search is not enabled")
        
        return self.semantic_service.search_similar(query_text, top_k=top_k)
    
    def extract_tables(self):
        """Delegate to existing table extraction."""
        return self.text_handler.extract_tables()
    
    def close(self):
        """Clean up resources."""
        if self.semantic_service:
            self.semantic_service.close()


def integrate_semantic_search_with_consumer(
    consumer_msg_processor,
    milvus_host: str = "localhost",
    milvus_port: str = "19530"
):
    """
    Factory function to create enhanced message processor with semantic search.
    
    This can be used to replace the existing msg_process method in your Kafka consumer.
    
    Args:
        consumer_msg_processor: Your existing message processor function
        milvus_host: Milvus server host
        milvus_port: Milvus server port
        
    Returns:
        Enhanced message processor function
    """
    semantic_service = SemanticSearchService(
        milvus_host=milvus_host,
        milvus_port=milvus_port
    )
    
    def enhanced_msg_process(msg):
        """
        Enhanced message processor that adds semantic indexing.
        """
        import json
        
        # Parse message
        json_string = msg.value().decode('utf-8')
        json_object = json.loads(json_string)
        
        document_id = json_object.get("document_id")
        document_path = json_object.get("document_path")
        found_toc = json_object.get("found_toc", False)
        
        # Import here to avoid circular imports
        from text_handler.TextService import TextHandler
        
        # Extract text
        text_handler = TextHandler(document_path)
        pages, paragraphs, sentences, toc = text_handler.extract_text(found_toc)
        
        # Index in Milvus for semantic search
        try:
            indexing_result = semantic_service.index_document(
                document_id=document_id,
                pages=pages,
                paragraphs=paragraphs,
                chunk_level="paragraph"
            )
            logger.info(f"Indexed {indexing_result['chunks_indexed']} chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to index document {document_id}: {e}")
        
        # Save to Elasticsearch (existing functionality)
        from elvira_elasticsearch_client import ElasticsearchClient
        client = ElasticsearchClient()
        client.save_extracted_text_to_elasticsearch(
            document_id=document_id,
            text_data=(pages, paragraphs, sentences, toc)
        )
        
        logger.info(f"Document {document_id} processed successfully")
    
    return enhanced_msg_process


if __name__ == "__main__":
    """
    Example: How to integrate with existing TextHandler
    """
    from text_handler.TextService import TextHandler
    
    # Example 1: Direct integration
    document_path = "src/test_data/algebra-a-diskretna-matematika.pdf"
    
    if os.path.exists(document_path):
        # Create regular text handler
        text_handler = TextHandler(document_path)
        
        # Wrap with semantic capabilities
        semantic_handler = SemanticTextHandler(
            text_handler=text_handler,
            milvus_host="localhost",
            milvus_port="19530",
            enable_semantic_search=True
        )
        
        # Extract and index
        result = semantic_handler.extract_and_index(
            document_id="algebra-001",
            found_toc=False,
            chunk_level="paragraph"
        )
        
        print(f"Processing result: {result}")
        
        # Search
        search_results = semantic_handler.search("lineárna algebra", top_k=3)
        print(f"\nSearch results:")
        for r in search_results:
            print(f"  Score: {r['score']:.4f}, Page: {r['page_num']}")
        
        # Cleanup
        semantic_handler.close()
    else:
        print(f"Test file not found: {document_path}")
