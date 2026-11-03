"""Paragraph and request embeddings (IP-018 Q16), one model instance per worker process."""

import logging
from functools import lru_cache
from typing import List

import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def model(name: str):
    from sentence_transformers import SentenceTransformer

    kwargs = {"device": settings.EMBEDDING_DEVICE, "trust_remote_code": True}
    if "arctic-embed" in name and settings.EMBEDDING_DEVICE == "cpu":
        # Arctic Embed v2.0's custom code defaults to xformers kernels that need a GPU.
        kwargs["config_kwargs"] = {"use_memory_efficient_attention": False, "unpad_inputs": False}
    logger.info("Loading embedding model %s", name)
    return SentenceTransformer(name, **kwargs)


def documents(texts: List[str], name: str = None) -> List[List[float]]:
    encoder = model(name or settings.EMBEDDING_MODEL)
    vectors = encoder.encode(
        texts, batch_size=settings.EMBEDDING_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
    )
    return [vector.tolist() for vector in vectors]


def query(text: str, name: str = None) -> List[float]:
    """A request, with the model's query prompt when it has one (Arctic Embed does)."""
    encoder = model(name or settings.EMBEDDING_MODEL)
    prompt = "query" if "query" in (getattr(encoder, "prompts", None) or {}) else None
    return encoder.encode(text, prompt_name=prompt, normalize_embeddings=True, show_progress_bar=False).tolist()
