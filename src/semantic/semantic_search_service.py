"""
Integrated Semantic Search Service
Combines text extraction, embedding generation, and vector storage for complete semantic search pipeline
"""

from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

from semantic_embeddings import SemanticEmbeddingsService, get_embeddings_service
from milvus_client import MilvusVectorStore, create_milvus_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSearchService:
    """
    Complete semantic search service for document processing and retrieval.
    
    Workflow:
    1. Extract text from documents (pages/paragraphs/sentences)
    2. Generate embeddings using XLM-R multilingual model
    3. Store embeddings in Milvus vector database
    4. Enable semantic similarity search
    5. Integrate with Elasticsearch for hybrid search (metadata + vectors)
    """
    
    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        model_cache_dir: Optional[str] = None,
        embedding_device: Optional[str] = None,
        batch_size: int = 32,
        collection_name: Optional[str] = None
    ):
        """
        Initialize semantic search service 
        
        Args:
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            model_cache_dir: Directory to cache embedding model
            embedding_device: Device for embeddings ('cuda', 'cpu', or None)
            batch_size: Batch size for embedding generation
            collection_name: Custom Milvus collection name
        """
        logger.info("Initializing Semantic Search Service")
        
        # Initialize embeddings service
        self.embeddings_service = get_embeddings_service(
            model_cache_dir=model_cache_dir,
            device=embedding_device,
            batch_size=batch_size
        )
        
        # Initialize vector store
        self.vector_store = create_milvus_client(
            host=milvus_host,
            port=milvus_port,
            collection_name=collection_name
        )
        
        logger.info("Semantic Search Service initialized successfully")
    
    def index_document(
        self,
        document_id: str,
        pages: List[Tuple[int, str]],
        paragraphs: List[List[str]],
        chunk_level: str = "paragraph"
    ) -> Dict:
        """
        Index a document by generating and storing embeddings.
        
        Args:
            document_id: Unique document identifier
            pages: List of (page_num, page_text) tuples
            paragraphs: List of paragraph lists (one list per page)
            chunk_level: Granularity level - "page" or "paragraph"
            
        Returns:
            Dictionary with indexing results
        """
        logger.info(f"Indexing document {document_id} at {chunk_level} level")
        
        # Generate embeddings
        embeddings, metadata = self.embeddings_service.encode_document_chunks(
            pages=pages,
            paragraphs=paragraphs,
            chunk_level=chunk_level
        )
        
        if len(embeddings) == 0:
            logger.warning(f"No embeddings generated for document {document_id}")
            return {
                "document_id": document_id,
                "success": False,
                "chunks_indexed": 0,
                "error": "No valid chunks found"
            }
        
        # Store embeddings in Milvus
        inserted_ids = self.vector_store.insert_embeddings(
            document_id=document_id,
            embeddings=embeddings,
            metadata=metadata
        )
        
        return {
            "document_id": document_id,
            "success": True,
            "chunks_indexed": len(inserted_ids),
            "chunk_level": chunk_level,
            "embedding_dimension": embeddings.shape[1] if len(embeddings) > 0 else 0
        }
    
    def search_similar(
        self,
        query_text: str,
        top_k: int = 10,
        document_id: Optional[str] = None,
        page_num: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for semantically similar chunks using query text.
        
        Args:
            query_text: Search query in any supported language
            top_k: Number of results to return
            document_id: Filter by specific document (optional)
            page_num: Filter by specific page (optional)
            
        Returns:
            List of search results with similarity scores
        """
        logger.info(f"Searching for: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embeddings_service.encode_text(query_text, normalize=True)[0]
        
        # Search in vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
            page_num=page_num
        )
        
        return results
    
    def search_by_selection(
        self,
        selected_text: str,
        top_k: int = 10,
        exclude_document_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Find similar content based on selected text snippet.
        Useful for "find similar" feature.
        
        Args:
            selected_text: Text selection from user
            top_k: Number of results to return
            exclude_document_id: Exclude results from this document
            
        Returns:
            List of similar chunks
        """
        logger.info(f"Finding similar content to selection: '{selected_text[:50]}...'")
        
        # Generate embedding for selected text
        selection_embedding = self.embeddings_service.encode_text(selected_text, normalize=True)[0]
        
        # Search (without document filter to find cross-document matches)
        results = self.vector_store.search(
            query_embedding=selection_embedding,
            top_k=top_k
        )
        
        # Filter out excluded document if specified
        if exclude_document_id:
            results = [r for r in results if r.get("document_id") != exclude_document_id]
        
        return results
    
    def delete_document(self, document_id: str) -> Dict:
        """
        Remove all embeddings for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Deletion result
        """
        logger.info(f"Deleting document: {document_id}")
        
        deleted_count = self.vector_store.delete_by_document_id(document_id)
        
        return {
            "document_id": document_id,
            "deleted_chunks": deleted_count,
            "success": deleted_count > 0
        }
    
    def get_service_info(self) -> Dict:
        """Get information about the service and its components."""
        return {
            "embeddings": self.embeddings_service.get_model_info(),
            "vector_store": self.vector_store.get_collection_stats(),
            "capabilities": {
                "multilingual": True,
                "languages_supported": "Slovak, English, German, Czech, and 45+ more",
                "chunk_levels": ["page", "paragraph"],
                "search_types": ["query_search", "selection_based_search"],
                "max_sequence_length": 512
            }
        }
    
    def close(self):
        """Clean up resources."""
        self.vector_store.close()
        logger.info("Semantic Search Service closed")


class HybridSearchService:
    """
    Hybrid search combining semantic search (Milvus) with keyword/metadata search (Elasticsearch).
    """
    
    def __init__(
        self,
        semantic_service: SemanticSearchService,
        elasticsearch_client  # Your existing ES client
    ):
        """
        Initialize hybrid search.
        
        Args:
            semantic_service: SemanticSearchService instance
            elasticsearch_client: Elasticsearch client instance
        """
        self.semantic_service = semantic_service
        self.es_client = elasticsearch_client
        logger.info("Hybrid Search Service initialized")
    
    def hybrid_search(
        self,
        query_text: str,
        top_k_semantic: int = 20,
        filters: Optional[Dict] = None,
        alpha: float = 0.7
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic similarity and keyword matching.
        
        Args:
            query_text: Search query
            top_k_semantic: Number of semantic results to retrieve
            filters: Optional filters (e.g., date range, document type)
            alpha: Weight for semantic vs keyword (0=all keyword, 1=all semantic)
            
        Returns:
            Ranked list of results combining both approaches
        """
        # 1. Semantic search
        semantic_results = self.semantic_service.search_similar(
            query_text=query_text,
            top_k=top_k_semantic
        )
        
        # 2. Elasticsearch keyword search (implement based on your ES schema)
        # es_results = self.es_client.search_documents(query_text, filters=filters)
        
        # 3. Score fusion and re-ranking
        # Combine scores: final_score = alpha * semantic_score + (1 - alpha) * keyword_score
        
        # For now, return semantic results (extend with ES integration)
        return semantic_results
    
    def search_with_filters(
        self,
        query_text: str,
        document_type: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        author: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search with metadata filters applied.
        
        Args:
            query_text: Search query
            document_type: Filter by document type
            date_range: Filter by date range (start, end)
            author: Filter by author
            top_k: Number of results
            
        Returns:
            Filtered and ranked results
        """
        # 1. Apply filters in Elasticsearch to get candidate document IDs
        # candidate_doc_ids = self.es_client.filter_documents(...)
        
        # 2. Perform semantic search within candidates
        # results = []
        # for doc_id in candidate_doc_ids:
        #     doc_results = self.semantic_service.search_similar(
        #         query_text, document_id=doc_id, top_k=top_k
        #     )
        #     results.extend(doc_results)
        
        # 3. Sort by relevance
        # results.sort(key=lambda x: x['score'], reverse=True)
        
        # For now, basic implementation
        return self.semantic_service.search_similar(query_text, top_k=top_k)


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Initialize service
    service = SemanticSearchService(
        milvus_host="localhost",
        milvus_port="19530"
    )
    
    # Example document data (simulated)
    doc_id = "algebra-a-diskretna-matematika"
    pages = [
        (1, "Algebra a diskrétna matematika sú základné oblasti matematiky."),
        (2, "Lineárna algebra študuje vektory a matice."),
        (3, "Diskrétna matematika zahŕňa teóriu grafov a kombinatoriku.")
    ]
    paragraphs = [
        ["Algebra a diskrétna matematika sú základné oblasti matematiky."],
        ["Lineárna algebra študuje vektory a matice."],
        ["Diskrétna matematika zahŕňa teóriu grafov a kombinatoriku."]
    ]
    
    # Index document
    result = service.index_document(doc_id, pages, paragraphs, chunk_level="paragraph")
    print(f"Indexing result: {result}")
    
    # Search
    query = "čo je lineárna algebra"
    results = service.search_similar(query, top_k=2)
    print(f"\nSearch results for '{query}':")
    for r in results:
        print(f"  Score: {r['score']:.4f}, Page: {r['page_num']}, Text: {r['text_preview']}")
    
    # Service info
    info = service.get_service_info()
    print(f"\nService info: {info}")
    
    # Cleanup
    service.close()
