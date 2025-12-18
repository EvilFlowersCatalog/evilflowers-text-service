from typing import List, Tuple, Dict
import logging

from semantic.EmbeddingGenerator import EmbeddingGenerator
from semantic.MilvusManager import MilvusManager
from config.semantic_config import SemanticConfig

logger = logging.getLogger(__name__)

class SemanticService:    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.milvus_manager = MilvusManager()
    
    def index_document(
        self,
        document_id: str,
        chunks: dict[list]
    ) -> Dict:
        # Generate embeddings
        embeddings = self._generate_embeddings(chunks, document_id)

        # Store in Milvus
        chunk_list = chunks['chunks']
        metadata = [chunk['metadata'] for chunk in chunk_list]
        return self._store_embeddings(document_id, embeddings, metadata)

    
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
        
    def _generate_embeddings(self, chunks, document_id=None):
        try:
            embeddings = self.embedding_generator.generate_embeddings(
                chunks,
                normalize=True,
                show_progress=True
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return {
                "document_id": document_id,
                "success": False,
                "chunks_indexed": 0,
                "error": str(e)
            }
        
        return embeddings
    
    def _store_embeddings(self, document_id, embeddings, metadata):
        try:
            inserted_ids = self.milvus_manager.insert_embeddings(
                document_id=document_id,
                embeddings=embeddings,
                metadata=metadata
            )
                        
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
