"""
Semantic search configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()


class SemanticConfig:
    """Configuration for semantic search components"""
    
    # Model Configuration
    MODEL_NAME = "sentence-transformers/stsb-xlm-r-multilingual"
    EMBEDDING_DIM = 768
    MAX_SEQUENCE_LENGTH = 512
    
    # Milvus Configuration
    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
    MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME', 'document_embeddings')
    
    # Processing Configuration
    BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
    DEVICE = os.getenv('EMBEDDING_DEVICE', 'cpu')  # 'cuda' or 'cpu'
    
    # Chunking Configuration
    CHUNK_LEVEL = os.getenv('CHUNK_LEVEL', 'paragraph')  # 'paragraph' or 'page'
    MIN_PARAGRAPH_LENGTH = 30  # Minimum words to index
    
    # Feature Flags
    SEMANTIC_SEARCH_ENABLED = os.getenv('SEMANTIC_SEARCH_ENABLED', 'true').lower() == 'true'
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        assert cls.EMBEDDING_DIM == 768, "XLM-R model requires 768 dimensions"
        assert cls.DEVICE in ['cpu', 'cuda'], "Device must be 'cpu' or 'cuda'"
        assert cls.CHUNK_LEVEL in ['paragraph', 'page'], "Chunk level must be 'paragraph' or 'page'"
        return True


# Validate on import
SemanticConfig.validate()