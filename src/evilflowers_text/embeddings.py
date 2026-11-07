"""Paragraph and request embeddings (IP-018 Q16), one model instance per worker process.

Callers depend on the `Encoder` protocol; `encoder(name)` returns the cached
sentence-transformers implementation. Another backend (a shared inference
server, say) is a new `Encoder`, not a change to the jobs.
"""

import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Protocol

from evilflowers_text.observability import metrics, span
from evilflowers_text.settings import get_settings

logger = logging.getLogger(__name__)

Vector = list[float]


class Encoder(Protocol):
    name: str

    def documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def query(self, text: str) -> Vector: ...


@contextmanager
def _measured(kind: str, model: str, inputs: int, **attributes) -> Iterator[None]:
    """Span `embeddings.<kind>` plus the duration and input-count metrics."""
    started = time.perf_counter()
    with span(f"embeddings.{kind}", **{"embedding.model": model, "embedding.inputs": inputs, **attributes}):
        yield
    metrics.record_embedding(kind, model, inputs, time.perf_counter() - started)


class SentenceTransformerEncoder:
    def __init__(self, name: str, *, device: str, batch_size: int):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.batch_size = batch_size
        kwargs: dict = {"device": device, "trust_remote_code": True}
        if "arctic-embed" in name and device == "cpu":
            # Arctic Embed v2.0's custom code defaults to xformers kernels that need a GPU.
            kwargs["config_kwargs"] = {"use_memory_efficient_attention": False, "unpad_inputs": False}
        logger.info("Loading embedding model %s", name)
        started = time.perf_counter()
        # Once per process and model (`encoder` caches): this span shows a cold start.
        with span("embeddings.model.load", **{"embedding.model": name, "embedding.device": device}):
            self._model = SentenceTransformer(name, **kwargs)
        metrics.MODEL_LOAD.record(time.perf_counter() - started, {"model": name})
        # Arctic Embed has a query prompt; models without one embed queries as they are.
        self._query_prompt = "query" if "query" in (getattr(self._model, "prompts", None) or {}) else None

    def documents(self, texts: Sequence[str]) -> list[Vector]:
        with _measured("documents", self.name, len(texts), **{"embedding.batch_size": self.batch_size}):
            vectors = self._model.encode(
                list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
            )
        return [vector.tolist() for vector in vectors]

    def query(self, text: str) -> Vector:
        with _measured("query", self.name, 1, **{"embedding.prompt": self._query_prompt}):
            vector = self._model.encode(
                text, prompt_name=self._query_prompt, normalize_embeddings=True, show_progress_bar=False
            )
        return vector.tolist()


def encoder(name: str | None = None) -> Encoder:
    """The encoder of `name` (default: `EMBEDDING_MODEL`), loaded once per process."""
    return _load(name or get_settings().embedding_model)


@lru_cache(maxsize=2)
def _load(name: str) -> Encoder:
    settings = get_settings()
    return SentenceTransformerEncoder(name, device=settings.embedding_device, batch_size=settings.embedding_batch_size)
