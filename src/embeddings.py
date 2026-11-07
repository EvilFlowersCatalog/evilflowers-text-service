"""Paragraph and request embeddings (IP-018 Q16), one model instance per worker process."""

import logging
import time
from functools import lru_cache
from typing import List

import settings
import telemetry

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def model(name: str):
    from sentence_transformers import SentenceTransformer

    kwargs = {"device": settings.EMBEDDING_DEVICE, "trust_remote_code": True}
    if "arctic-embed" in name and settings.EMBEDDING_DEVICE == "cpu":
        # Arctic Embed v2.0's custom code defaults to xformers kernels that need a GPU.
        kwargs["config_kwargs"] = {"use_memory_efficient_attention": False, "unpad_inputs": False}
    logger.info("Loading embedding model %s", name)
    started = time.perf_counter()
    # Only the first call per process and model gets here (lru_cache): this span shows a cold start.
    with telemetry.span("embeddings.model.load", **{"embedding.model": name, "embedding.device": settings.EMBEDDING_DEVICE}):
        encoder = SentenceTransformer(name, **kwargs)
    telemetry.record_model_load(name, time.perf_counter() - started)
    return encoder


def documents(texts: List[str], name: str = None) -> List[List[float]]:
    name = name or settings.EMBEDDING_MODEL
    encoder = model(name)
    started = time.perf_counter()
    with telemetry.span(
        "embeddings.documents",
        **{"embedding.model": name, "embedding.inputs": len(texts), "embedding.batch_size": settings.EMBEDDING_BATCH_SIZE},
    ):
        vectors = encoder.encode(
            texts, batch_size=settings.EMBEDDING_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
        )
    telemetry.record_embedding("documents", name, len(texts), time.perf_counter() - started)
    return [vector.tolist() for vector in vectors]


def query(text: str, name: str = None) -> List[float]:
    """A request, with the model's query prompt when it has one (Arctic Embed does)."""
    name = name or settings.EMBEDDING_MODEL
    encoder = model(name)
    prompt = "query" if "query" in (getattr(encoder, "prompts", None) or {}) else None
    started = time.perf_counter()
    with telemetry.span("embeddings.query", **{"embedding.model": name, "embedding.prompt": prompt}):
        vector = encoder.encode(text, prompt_name=prompt, normalize_embeddings=True, show_progress_bar=False)
    telemetry.record_embedding("query", name, 1, time.perf_counter() - started)
    return vector.tolist()
