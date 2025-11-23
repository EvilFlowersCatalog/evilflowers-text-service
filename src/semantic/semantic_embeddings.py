"""
Semantic Embeddings Service for Multilingual Semantic Search
Uses sentence-transformers/stsb-xlm-r-multilingual model for Slovak and multilingual support
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticEmbeddingsService:
    """
    Service for generating semantic embeddings using XLM-R based multilingual model.
    Optimized for Slovak and other languages with proper semantic alignment.
    
    Model: sentence-transformers/stsb-xlm-r-multilingual
    - Trained on multilingual STS (Semantic Textual Similarity) and NLI
    - Supports 50+ languages including Slovak
    - Outputs 768-dimensional embeddings
    - Good for sentences and paragraphs (≤512 tokens)
    """
    
    MODEL_NAME = "sentence-transformers/stsb-xlm-r-multilingual"
    EMBEDDING_DIM = 768
    MAX_SEQUENCE_LENGTH = 512  # XLM-R token limit
    
    def __init__(
        self, 
        model_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize the semantic embeddings service.
        
        Args:
            model_cache_dir: Directory to cache the model (default: ~/.cache/huggingface)
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for encoding (larger = faster but more memory)
        """
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_cache_dir = model_cache_dir
        
        logger.info(f"Initializing SemanticEmbeddingsService on device: {self.device}")
        self._load_model()
        
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading model: {self.MODEL_NAME}")
            self.model = SentenceTransformer(
                self.MODEL_NAME,
                cache_folder=self.model_cache_dir,
                device=self.device
            )
            logger.info(f"Model loaded successfully. Embedding dimension: {self.EMBEDDING_DIM}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def encode_text(
        self, 
        texts: Union[str, List[str]], 
        normalize: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into semantic embeddings.
        
        Args:
            texts: Single text string or list of text strings
            normalize: Whether to normalize embeddings to unit length (recommended for cosine similarity)
            show_progress_bar: Show progress bar for batch encoding
            
        Returns:
            numpy array of shape (n_texts, 768) containing embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([])
        
        # Encode with model
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            device=self.device
        )
        
        return embeddings
    
    def encode_paragraphs(
        self, 
        paragraphs: List[str],
        max_length: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode paragraphs with optional truncation.
        
        Args:
            paragraphs: List of paragraph texts
            max_length: Maximum length in tokens (default: 512)
            normalize: Whether to normalize embeddings
            
        Returns:
            numpy array of embeddings
        """
        max_length = max_length or self.MAX_SEQUENCE_LENGTH
        
        # Filter out empty paragraphs
        valid_paragraphs = [p for p in paragraphs if p and p.strip()]
        
        if not valid_paragraphs:
            return np.array([])
        
        logger.info(f"Encoding {len(valid_paragraphs)} paragraphs")
        return self.encode_text(valid_paragraphs, normalize=normalize, show_progress_bar=True)
    
    def encode_document_chunks(
        self,
        pages: List[tuple],  # [(page_num, page_text), ...]
        paragraphs: List[List[str]],  # List of paragraphs per page
        chunk_level: str = "paragraph"  # "page" or "paragraph"
    ) -> tuple:
        """
        Encode document chunks at different granularities.
        
        Args:
            pages: List of (page_num, page_text) tuples
            paragraphs: List of paragraph lists (one list per page)
            chunk_level: Level of chunking - "page" or "paragraph"
            
        Returns:
            Tuple of (embeddings, metadata) where metadata contains chunk information
        """
        embeddings_list = []
        metadata_list = []
        
        if chunk_level == "page":
            # Encode entire pages
            for page_num, page_text in pages:
                if page_text and page_text.strip():
                    emb = self.encode_text(page_text, normalize=True)
                    embeddings_list.append(emb[0])
                    metadata_list.append({
                        "page_num": page_num,
                        "chunk_type": "page",
                        "text_length": len(page_text)
                    })
        
        elif chunk_level == "paragraph":
            # Encode paragraphs with page tracking
            for page_idx, page_paragraphs in enumerate(paragraphs):
                page_num = page_idx + 1
                for para_idx, para_text in enumerate(page_paragraphs):
                    if para_text and para_text.strip() and len(para_text) > 20:
                        emb = self.encode_text(para_text, normalize=True)
                        embeddings_list.append(emb[0])
                        metadata_list.append({
                            "page_num": page_num,
                            "paragraph_idx": para_idx,
                            "chunk_type": "paragraph",
                            "text_length": len(para_text),
                            "text_preview": para_text[:100]
                        })
        
        embeddings = np.array(embeddings_list) if embeddings_list else np.array([])
        
        logger.info(f"Generated {len(embeddings)} embeddings at {chunk_level} level")
        return embeddings, metadata_list
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Single embedding vector (768,)
            document_embeddings: Matrix of document embeddings (n, 768)
            
        Returns:
            Array of similarity scores (n,)
        """
        # Ensure embeddings are normalized
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(doc_norms, query_norm)
        return similarities
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.MODEL_NAME,
            "embedding_dimension": self.EMBEDDING_DIM,
            "max_sequence_length": self.MAX_SEQUENCE_LENGTH,
            "device": self.device,
            "batch_size": self.batch_size,
            "supports_languages": "50+ languages including Slovak, English, German, etc.",
            "use_case": "Semantic similarity, cross-lingual search, document retrieval"
        }


# Singleton instance for reuse
_global_embeddings_service: Optional[SemanticEmbeddingsService] = None


def get_embeddings_service(
    model_cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 32
) -> SemanticEmbeddingsService:
    """
    Get or create a global embeddings service instance (singleton pattern).
    
    Args:
        model_cache_dir: Directory to cache the model
        device: Device to run model on
        batch_size: Batch size for encoding
        
    Returns:
        SemanticEmbeddingsService instance
    """
    global _global_embeddings_service
    
    if _global_embeddings_service is None:
        _global_embeddings_service = SemanticEmbeddingsService(
            model_cache_dir=model_cache_dir,
            device=device,
            batch_size=batch_size
        )
    
    return _global_embeddings_service


if __name__ == "__main__":
    # Example usage
    service = SemanticEmbeddingsService()
    
    # Test with Slovak and English text
    test_texts = [
        "Univerzitná knižnica v Bratislave",
        "University library in Bratislava",
        "Sémantické vyhľadávanie dokumentov",
        "Semantic document search"
    ]
    
    print("Testing semantic embeddings service...")
    embeddings = service.encode_text(test_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Compute similarities
    query_emb = embeddings[0]  # Slovak: "Univerzitná knižnica v Bratislave"
    similarities = service.compute_similarity(query_emb, embeddings)
    
    print("\nSimilarities to query 'Univerzitná knižnica v Bratislave':")
    for text, sim in zip(test_texts, similarities):
        print(f"  {text[:50]}: {sim:.4f}")
    
    print(f"\nModel info: {service.get_model_info()}")
