import os
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from typing import List, Dict, Optional
import numpy as np
import logging
from datetime import datetime

from config.semantic_config import SemanticConfig

logger = logging.getLogger(__name__)


class MilvusManager:
    """
    Manages Milvus vector database operations.
    Singleton pattern for connection reuse.
    """
    
    _instance = None
    _collection = None
    _connected = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._connected:
            self._connect()
            self._setup_collection()
    
    def _connect(self):
        """Connect to Milvus server or use embedded Milvus Lite"""
        
        # Check if we should use Milvus Lite (embedded)
        use_lite = os.getenv('USE_MILVUS_LITE', 'false').lower() == 'true'
        
        if use_lite:
            # Use embedded Milvus Lite (no server needed!)
            logger.info("Using Milvus Lite (embedded mode)")
            try:
                db_file = os.getenv('MILVUS_LITE_DB', './milvus_lite.db')
                
                # For Milvus Lite, use URI connection
                connections.connect(
                    alias="default",
                    uri=db_file  # This triggers Milvus Lite mode
                )
                
                self._connected = True
                logger.info(f"✓ Connected to Milvus Lite: {db_file}")
                
            except Exception as e:
                logger.error(f"Failed to connect to Milvus Lite: {e}")
                raise
        else:
            # Use regular Milvus server
            try:
                logger.info(f"Connecting to Milvus server at {SemanticConfig.MILVUS_HOST}:{SemanticConfig.MILVUS_PORT}")
                
                connections.connect(
                    alias="default",
                    host=SemanticConfig.MILVUS_HOST,
                    port=SemanticConfig.MILVUS_PORT
                )
                
                self._connected = True
                logger.info("✓ Connected to Milvus server")
                
            except Exception as e:
                logger.error(f"Failed to connect to Milvus: {e}")
                raise
    
    def _setup_collection(self):
        """Create or load collection"""
        collection_name = SemanticConfig.MILVUS_COLLECTION_NAME
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="source_page", dtype=DataType.INT64),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="word_count", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=SemanticConfig.EMBEDDING_DIM),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Document embeddings for semantic search"
        )
        
        # Create or load collection
        if utility.has_collection(collection_name):
            logger.info(f"Loading existing collection: {collection_name}")
            self._collection = Collection(name=collection_name)
        else:
            logger.info(f"Creating new collection: {collection_name}")
            self._collection = Collection(
                name=collection_name,
                schema=schema
            )
            self._create_index()
        
        # Load collection into memory
        self._collection.load()
        logger.info(f"✓ Collection {collection_name} is ready")
    
    def _create_index(self):
        """Create index for fast similarity search"""
        
        # Check if using Milvus Lite
        use_lite = os.getenv('USE_MILVUS_LITE', 'false').lower() == 'true'
        
        if use_lite:
            # Milvus Lite: Use AUTOINDEX (simpler, automatically optimized)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "AUTOINDEX",
                "params": {}
            }
            logger.info("Creating AUTOINDEX for Milvus Lite...")
        else:
            # Milvus Server: Use HNSW (faster for large datasets)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 256
                }
            }
            logger.info("Creating HNSW index for Milvus Server...")
        
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        logger.info("✓ Index created")
    
    def insert_embeddings(
        self,
        document_id: str,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ) -> List[int]:
        """
        Insert embeddings with metadata.
        
        Args:
            document_id: Document identifier
            embeddings: Array of embeddings (n, 768)
            metadata: List of metadata dicts
            
        Returns:
            List of inserted IDs
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match metadata entries")
        
        if len(embeddings) == 0:
            logger.warning("No embeddings to insert")
            return []
        
        current_time = datetime.utcnow().isoformat()
        
        # Prepare data
        entities = [
            [document_id] * len(embeddings),                          # document_id
            [m.get("source_page", -1) for m in metadata],             # page_num
            [m.get("section", "content") for m in metadata],          # chunk_type
            [m.get("chunk_index", 0) for m in metadata],              # paragraph_idx
            [m.get("text", "")[:2000] for m in metadata],             # text
            [m.get("word_count", 0) for m in metadata],               # text_length
            embeddings.tolist(),                                       # embedding
            [current_time] * len(embeddings)                          # created_at
        ]
        
        # Insert
        logger.info(f"Inserting {len(embeddings)} embeddings for document {document_id}")
        
        try:
            insert_result = self._collection.insert(entities)
            self._collection.flush()
            
            logger.info(f"✓ Inserted {len(insert_result.primary_keys)} embeddings")
            return insert_result.primary_keys
            
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        document_id: Optional[str] = None,
        page_num: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query vector (768,)
            top_k: Number of results
            document_id: Filter by document (optional)
            page_num: Filter by page (optional)
            
        Returns:
            List of search results
        """
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 128}
        }
        
        # Build filter
        filter_expr = None
        if document_id and page_num is not None:
            filter_expr = f'document_id == "{document_id}" && page_num == {page_num}'
        elif document_id:
            filter_expr = f'document_id == "{document_id}"'
        elif page_num is not None:
            filter_expr = f'page_num == {page_num}'
        
        output_fields = [
            "document_id", "source_page", "section",
            "chunk_index", "text", "word_count"
        ]
        
        try:
            logger.debug(f"Searching for top {top_k} results")
            
            results = self._collection.search(
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
                    for field in output_fields:
                        result[field] = hit.entity.get(field)
                    formatted_results.append(result)
            
            logger.debug(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all embeddings for a document"""
        expr = f'document_id == "{document_id}"'
        
        try:
            logger.info(f"Deleting embeddings for document: {document_id}")
            result = self._collection.delete(expr)
            self._collection.flush()
            
            deleted_count = result.delete_count
            logger.info(f"✓ Deleted {deleted_count} embeddings")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            "collection_name": SemanticConfig.MILVUS_COLLECTION_NAME,
            "total_entities": self._collection.num_entities,
            "embedding_dim": SemanticConfig.EMBEDDING_DIM
        }
    
    def close(self):
        """Release resources"""
        if self._collection:
            self._collection.release()
        connections.disconnect("default")
        self._connected = False
        logger.info("Disconnected from Milvus")