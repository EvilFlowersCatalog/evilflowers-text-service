"""Paragraph embeddings through the shared `tei-embed` (Text Embeddings Inference).

The client contract is the knowledge base's (`Proposals/Search-Embeddings`,
`Cookbooks/Shared-Inference`):

- paragraphs are documents: plain text, no `Instruct:` prefix (that is for queries,
  which the catalog embeds itself);
- at most `max_client_batch_size` (32) inputs per request;
- a paragraph longer than the model's `max_input_length` (512 tokens) is split at
  sentence boundaries and its pieces' vectors averaged, instead of letting TEI
  truncate it (`truncate` stays on as the backstop);
- 60 s per batch, retries with backoff on 429/503 (TEI refuses work above its
  concurrency limit) and on connection errors;
- the model's identity (`model_id@model_sha` from `/info`) travels with the vectors.

Callers depend on the `Encoder` protocol; the jobs never see HTTP.
"""

import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

import numpy as np
import requests

from evilflowers_text.clients.catalog import batches
from evilflowers_text.clients.http import session
from evilflowers_text.extraction.paragraphs import sentences
from evilflowers_text.observability import metrics, span
from evilflowers_text.settings import Settings, get_settings

logger = logging.getLogger(__name__)

Vector = list[float]

#: TEI answers these when it is over its concurrency limit or not ready yet.
RETRIED_STATUSES = frozenset({429, 503})


class Encoder(Protocol):
    @property
    def name(self) -> str:
        """The model's identity, recorded in the run's provenance."""
        ...

    def documents(self, texts: Sequence[str]) -> list[Vector]: ...


@contextmanager
def _measured(kind: str, model: str, inputs: int, **attributes) -> Iterator[None]:
    """Span `embeddings.<kind>` plus the duration and input-count metrics."""
    started = time.perf_counter()
    with span(f"embeddings.{kind}", **{"embedding.model": model, "embedding.inputs": inputs, **attributes}):
        yield
    metrics.record_embedding(kind, model, inputs, time.perf_counter() - started)


def _char_offsets(text: str) -> dict[int, int]:
    """UTF-8 byte offset → character offset, at every character boundary (TEI's token offsets are bytes)."""
    offsets: dict[int, int] = {}
    position = 0
    for index, char in enumerate(text):
        offsets[position] = index
        position += len(char.encode())
    offsets[position] = len(text)
    return offsets


def _token_starts(text: str, tokens: Sequence[dict[str, Any]]) -> list[int]:
    """Character offset of every non-special token."""
    offsets = _char_offsets(text)
    return [
        offsets[token["start"]]
        for token in tokens
        if not token["special"] and token.get("start") is not None and token["start"] in offsets
    ]


def _window_end(starts: Sequence[int], first: int, budget: int, sentence_starts: set[int]) -> int:
    """Index of the token after the window from `first`: at most `budget` tokens, ending before a sentence if any."""
    last = min(first + budget, len(starts))
    if last == len(starts):
        return last
    return next((index for index in range(last, first, -1) if starts[index] in sentence_starts), last)


def split(text: str, tokens: Sequence[dict[str, Any]], limit: int) -> list[str]:
    """`text` in pieces of at most `limit` tokens (special tokens included), cut at sentence starts when possible.

    `tokens` is TEI's `/tokenize` output for `text`, with special tokens.
    """
    if len(tokens) <= limit:
        return [text]
    budget = max(1, limit - sum(1 for token in tokens if token["special"]))
    starts = _token_starts(text, tokens)
    sentence_starts = {start for start, _end in sentences(text)}
    pieces: list[str] = []
    first = 0
    while first < len(starts):
        last = _window_end(starts, first, budget, sentence_starts)
        end = starts[last] if last < len(starts) else len(text)
        if piece := text[starts[first] : end].strip():
            pieces.append(piece)
        first = last
    return pieces or [text]


class TeiEncoder:
    """`POST /embed` on a TEI server; one instance per process (`encoder()`)."""

    def __init__(self, *, settings: Settings | None = None, http: requests.Session | None = None):
        self.settings = settings or get_settings()
        self.base = self.settings.tei_url.rstrip("/")
        self.http = http or session(self.settings.tei_api_key)
        if self.settings.tei_host:
            # Compose on gateway reaches the cluster's gateway by IP and routes by Host (Known-Issues #7).
            self.http.headers["Host"] = self.settings.tei_host
        self._info: dict[str, Any] | None = None

    def _call(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        """One request, retried on 429/503 and connection errors with exponential backoff."""
        attempts = self.settings.tei_attempts
        for attempt in range(1, attempts + 1):
            try:
                response = self.http.request(
                    method, f"{self.base}{path}", json=payload, timeout=self.settings.tei_timeout
                )
            except requests.RequestException:
                if attempt == attempts:
                    raise
                logger.warning("tei-embed unreachable at %s%s, retrying", self.base, path)
            else:
                if response.status_code not in RETRIED_STATUSES or attempt == attempts:
                    response.raise_for_status()
                    return response.json()
                logger.warning("tei-embed answered %s on %s, retrying", response.status_code, path)
            time.sleep(2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    @property
    def info(self) -> dict[str, Any]:
        """`GET /info`, once per process: the served model and its limits."""
        if self._info is None:
            self._info = self._call("GET", "/info")
        return self._info

    @property
    def name(self) -> str:
        info = self.info
        return f"{info['model_id']}@{info['model_sha']}" if info.get("model_sha") else info["model_id"]

    @property
    def batch_size(self) -> int:
        configured = self.settings.tei_batch_size
        return min(configured, self.info.get("max_client_batch_size") or configured)

    def _batches(self, texts: Sequence[str]) -> Iterator[list[str]]:
        return (list(batch) for batch in batches(texts, self.batch_size))

    def _pieces(self, texts: Sequence[str]) -> list[list[str]]:
        """Every text as the pieces that fit the model's input length (usually one)."""
        limit = self.info["max_input_length"]
        pieces: list[list[str]] = []
        for batch in self._batches(texts):
            tokenized = self._call("POST", "/tokenize", {"inputs": batch, "add_special_tokens": True})
            pieces.extend(split(text, tokens, limit) for text, tokens in zip(batch, tokenized, strict=True))
        return pieces

    def _embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for batch in self._batches(texts):
            with _measured("documents", self.name, len(batch), **{"embedding.batch_size": self.batch_size}):
                vectors.extend(self._call("POST", "/embed", {"inputs": batch, "normalize": True, "truncate": True}))
        return vectors

    def documents(self, texts: Sequence[str]) -> list[Vector]:
        """One normalized vector per text; a split text gets the normalized mean of its pieces."""
        if not texts:
            return []
        pieces = self._pieces(texts)
        split_count = sum(1 for parts in pieces if len(parts) > 1)
        if split_count:
            logger.info("Split %d of %d texts longer than the model's input length", split_count, len(texts))
        flat = self._embed([piece for parts in pieces for piece in parts])
        vectors: list[Vector] = []
        position = 0
        for parts in pieces:
            group = flat[position : position + len(parts)]
            position += len(parts)
            if len(group) == 1:
                vectors.append(group[0])
            else:
                mean = np.mean(np.asarray(group, dtype=np.float64), axis=0)
                vectors.append((mean / (np.linalg.norm(mean) or 1.0)).tolist())
        return vectors


_encoder: TeiEncoder | None = None


def encoder() -> Encoder:
    """The process's TEI encoder (its `/info` is read once)."""
    global _encoder
    if _encoder is None:
        _encoder = TeiEncoder()
    return _encoder
