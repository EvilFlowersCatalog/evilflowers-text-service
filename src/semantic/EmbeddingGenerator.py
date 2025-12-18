import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging
from config.Config import Config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers XLM-R model.
    Singleton pattern for model reuse.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading model: {Config.MODEL_NAME}")
            logger.info(f"Using device: {Config.DEVICE}")
            
            self._model = SentenceTransformer(
                Config.MODEL_NAME,
                device=Config.DEVICE
            )
            
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(
        self, 
        chunks: dict[list],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (n_texts, 768)
        """

        chunk_list = chunks['chunks']
        texts = [chunk['text'] for chunk in chunk_list]

        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            
            embeddings = self._model.encode(
                texts,
                batch_size=Config.BATCH_SIZE,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                device=Config.DEVICE
            )
            
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text"""
        embeddings = self.generate_embeddings([text], normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else np.array([])
        