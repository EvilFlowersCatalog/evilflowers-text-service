"""
Milvus Vector Database Client for Semantic Search
Handles vector storage, indexing, and similarity search operations
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """
    Client for interacting with Milvus vector database.
    Optimized for storing document embeddings and performing semantic similarity search.
    """
    
    COLLECTION_NAME = "document_embeddings"
    EMBEDDING_DIM = 768  # XLM-R multilingual model dimension
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        collection_name: Optional[str] = None
    ):
        """
        Initialize Milvus client.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            user: Username for authentication (if required)
            password: Password for authentication (if required)
            collection_name: Custom collection name (default: document_embeddings)
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name or self.COLLECTION_NAME
        self.collection = None
        
        self._connect(user, password)
        self._setup_collection()
    
    def _connect(self, user: str, password: str):
        """Connect to Milvus server."""
        try:
            logger.info(f"Connecting to Milvus at {self.host}:{self.port}")
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=user,
                password=password
            )
            logger.info("Successfully connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _setup_collection(self):
        """Create or load collection with appropriate schema."""
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="paragraph_idx", dtype=DataType.INT64),
            FieldSchema(name="text_preview", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="text_length", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.EMBEDDING_DIM),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Document embeddings for semantic search"
        )
        
        # Create or load collection
        if utility.has_collection(self.collection_name):
            logger.info(f"Loading existing collection: {self.collection_name}")
            self.collection = Collection(name=self.collection_name)
        else:
            logger.info(f"Creating new collection: {self.collection_name}")
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            self._create_index()
        
        # Load collection into memory
        self.collection.load()
        logger.info(f"Collection {self.collection_name} is ready")
    
    def _create_index(self):
        """
        Create HNSW index for fast approximate nearest neighbor search.
        HNSW (Hierarchical Navigable Small World) is efficient for high-dimensional vectors.
        """
        index_params = {
            "metric_type": "COSINE",  # Cosine similarity
            "index_type": "HNSW",
            "params": {
                "M": 16,  # Number of connections per layer
                "efConstruction": 256  # Size of dynamic candidate list for construction
            }
        }
        
        logger.info("Creating HNSW index on embedding field")
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        logger.info("Index created successfully")
    
    def insert_embeddings(
        self,
        document_id: str,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ) -> List[int]:
        """
        Insert document embeddings with metadata.
        
        Args:
            document_id: Unique document identifier
            embeddings: Array of embeddings (n, 768)
            metadata: List of metadata dicts, one per embedding
            
        Returns:
            List of inserted IDs
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        if len(embeddings) == 0:
            logger.warning("No embeddings to insert")
            return []
        
        # Prepare data for insertion
        current_time = datetime.utcnow().isoformat()
        
        entities = [
            [document_id] * len(embeddings),  # document_id
            [m.get("page_num", 0) for m in metadata],  # page_num
            [m.get("chunk_type", "unknown") for m in metadata],  # chunk_type
            [m.get("paragraph_idx", -1) for m in metadata],  # paragraph_idx
            [m.get("text_preview", "")[:500] for m in metadata],  # text_preview
            [m.get("text_length", 0) for m in metadata],  # text_length
            embeddings.tolist(),  # embedding
            [current_time] * len(embeddings)  # created_at
        ]
        
        # Insert data
        logger.info(f"Inserting {len(embeddings)} embeddings for document {document_id}")
        insert_result = self.collection.insert(entities)
        
        # Flush to persist data
        self.collection.flush()
        
        logger.info(f"Successfully inserted {len(insert_result.primary_keys)} embeddings")
        return insert_result.primary_keys
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        document_id: Optional[str] = None,
        page_num: Optional[int] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for similar embeddings using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector (768,)
            top_k: Number of results to return
            document_id: Filter by specific document (optional)
            page_num: Filter by specific page (optional)
            output_fields: Fields to return in results
            
        Returns:
            List of search results with scores and metadata
        """
        # Prepare search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 128}  # Size of dynamic candidate list for search
        }
        
        # Build filter expression
        filter_expr = None
        if document_id and page_num is not None:
            filter_expr = f'document_id == "{document_id}" && page_num == {page_num}'
        elif document_id:
            filter_expr = f'document_id == "{document_id}"'
        elif page_num is not None:
            filter_expr = f'page_num == {page_num}'
        
        # Default output fields
        if output_fields is None:
            output_fields = [
                "document_id", "page_num", "chunk_type", 
                "paragraph_idx", "text_preview", "text_length"
            ]
        
        # Perform search
        logger.info(f"Searching for top {top_k} similar embeddings")
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "distance": hit.distance,
                }
                # Add output fields
                for field in output_fields:
                    result[field] = hit.entity.get(field)
                formatted_results.append(result)
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all embeddings for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of deleted entities
        """
        expr = f'document_id == "{document_id}"'
        logger.info(f"Deleting embeddings for document: {document_id}")
        
        result = self.collection.delete(expr)
        self.collection.flush()
        
        deleted_count = result.delete_count
        logger.info(f"Deleted {deleted_count} embeddings")
        return deleted_count
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        stats = self.collection.num_entities
        
        return {
            "collection_name": self.collection_name,
            "total_entities": stats,
            "embedding_dimension": self.EMBEDDING_DIM
        }
    
    def close(self):
        """Release collection and disconnect."""
        if self.collection:
            self.collection.release()
        connections.disconnect("default")
        logger.info("Disconnected from Milvus")


def create_milvus_client(
    host: str = "localhost",
    port: str = "19530",
    collection_name: Optional[str] = None
) -> MilvusVectorStore:
    """
    Factory function to create Milvus client.
    
    Args:
        host: Milvus server host
        port: Milvus server port
        collection_name: Custom collection name
        
    Returns:
        MilvusVectorStore instance
    """
    return MilvusVectorStore(
        host=host,
        port=port,
        collection_name=collection_name
    )


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create client
    client = MilvusVectorStore(host="localhost", port="19530")
    
    # Example: Insert dummy embeddings
    doc_id = "test_document_001"
    dummy_embeddings = np.random.rand(5, 768).astype(np.float32)
    dummy_metadata = [
        {"page_num": 1, "chunk_type": "paragraph", "paragraph_idx": 0, "text_preview": "First paragraph", "text_length": 150},
        {"page_num": 1, "chunk_type": "paragraph", "paragraph_idx": 1, "text_preview": "Second paragraph", "text_length": 200},
        {"page_num": 2, "chunk_type": "paragraph", "paragraph_idx": 0, "text_preview": "Third paragraph", "text_length": 180},
        {"page_num": 2, "chunk_type": "paragraph", "paragraph_idx": 1, "text_preview": "Fourth paragraph", "text_length": 220},
        {"page_num": 3, "chunk_type": "paragraph", "paragraph_idx": 0, "text_preview": "Fifth paragraph", "text_length": 190},
    ]
    
    # Insert
    ids = client.insert_embeddings(doc_id, dummy_embeddings, dummy_metadata)
    print(f"Inserted IDs: {ids}")
    
    # Search
    query_emb = np.random.rand(768).astype(np.float32)
    results = client.search(query_emb, top_k=3)
    print(f"\nSearch results:")
    for r in results:
        print(f"  Score: {r['score']:.4f}, Page: {r['page_num']}, Preview: {r['text_preview']}")
    
    # Stats
    stats = client.get_collection_stats()
    print(f"\nCollection stats: {stats}")
    
    # Cleanup
    client.close()
