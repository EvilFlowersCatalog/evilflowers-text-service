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
- the model's identity (`model_id@model_sha` from `/info`) travels with the vectors;
- `TEI_LANE`, when set, goes out as `x-embed-lane` on every request, and the router
  answers with the lane that served it (knowledge base, `Proposals/Embed-Lanes`).

Up to `TEI_CONCURRENCY` batches are in flight at once (issue #11). With more than one,
a text batch's pieces are embedded as soon as its `/tokenize` is back, rather than after
the whole run is tokenized. The `/embed` batches are cut from the pieces in input order
whatever the concurrency, so the requests and the vectors are the same; with one, the
order of the requests is today's too: every `/tokenize`, then every `/embed`.

Tracing: one `embeddings.documents` span per call, with the batch count, retries,
slowest batch, concurrency and serving lane as attributes; the requests inside it are not
traced (a book is hundreds of batches, and per-batch spans pushed one task past Tempo's
per-trace limit, issue #10). Per-batch timings are the `text.embedding.duration` histogram.

Callers depend on the `Encoder` protocol; the jobs never see HTTP.
"""

import contextvars
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import requests
from requests.adapters import DEFAULT_POOLSIZE, HTTPAdapter

from evilflowers_text.clients.catalog import batches
from evilflowers_text.clients.http import session
from evilflowers_text.extraction.paragraphs import sentences
from evilflowers_text.observability import annotate, metrics, span, untraced_http
from evilflowers_text.settings import Settings, get_settings

logger = logging.getLogger(__name__)

Vector = list[float]

#: TEI answers these when it is over its concurrency limit or not ready yet.
RETRIED_STATUSES = frozenset({429, 503})

#: Request: the lane asked for. Response: the lane that served the request.
LANE_HEADER = "x-embed-lane"


class Encoder(Protocol):
    @property
    def name(self) -> str:
        """The model's identity, recorded in the run's provenance."""
        ...

    def documents(self, texts: Sequence[str]) -> list[Vector]: ...


@dataclass
class _Tally:
    """One `documents()` call, for its span: a book is hundreds of batches, so they are not spans of their own.

    Batches in flight update it from several threads, through the methods below.
    """

    concurrency: int = 1
    inputs: int = 0
    batches: int = 0
    retries: int = 0
    max_batch_seconds: float = 0.0
    lanes: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def retried(self) -> None:
        with self._lock:
            self.retries += 1

    def served(self, lane: str | None) -> None:
        if lane:
            with self._lock:
                self.lanes.add(lane)

    def embedded(self, inputs: int, seconds: float) -> None:
        with self._lock:
            self.inputs += inputs
            self.batches += 1
            self.max_batch_seconds = max(self.max_batch_seconds, seconds)

    def attributes(self) -> dict[str, Any]:
        with self._lock:
            return {
                "embedding.inputs": self.inputs,
                "embedding.batches": self.batches,
                "tei.retries": self.retries,
                "tei.max_batch_ms": round(self.max_batch_seconds * 1000),
                "tei.concurrency": self.concurrency,
                # Normally one lane; several only if the router moved the run between lanes.
                "tei.lane": ",".join(sorted(self.lanes)) or None,
            }


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


class _Schedule:
    """The order a `documents()` call's requests may go out in, and their answers put back in input order.

    The `/embed` batches are cut from the pieces in input order as the `/tokenize` answers
    come back, so they are the same batches whatever order TEI answers in. With
    `prefer_embed` (more than one batch in flight) a ready `/embed` goes before the next
    `/tokenize`, which gets vectors moving early; without it every `/tokenize` goes first,
    the order of the sequential client.
    """

    def __init__(self, text_batches: list[list[str]], size: int, *, prefer_embed: bool):
        self.text_batches = text_batches
        self.size = size
        self.prefer_embed = prefer_embed
        self.submitted = 0  # /tokenize batches sent
        self.folded = 0  # /tokenize answers turned into pieces, in order
        self.tokenized: dict[int, list[list[str]]] = {}  # answers waiting for an earlier batch's
        self.pieces: list[list[str]] = []
        self.pending: list[str] = []  # pieces not in an /embed batch yet
        self.ready: deque[tuple[int, list[str]]] = deque()  # /embed batches not sent yet, numbered
        self.embed_batches = 0
        self.vectors: dict[int, list[Vector]] = {}

    def next(self) -> tuple[str, int, list[str]] | None:
        """The next request to send, or None until an answer makes one ready."""
        can_tokenize = self.submitted < len(self.text_batches)
        if self.ready and (self.prefer_embed or not can_tokenize):
            return ("embed", *self.ready.popleft())
        if can_tokenize:
            self.submitted += 1
            return "tokenize", self.submitted - 1, self.text_batches[self.submitted - 1]
        return None

    def answered(self, kind: str, index: int, result: Any) -> None:
        if kind == "embed":
            self.vectors[index] = result
            return
        self.tokenized[index] = result
        while self.folded in self.tokenized:
            for parts in self.tokenized.pop(self.folded):
                self.pieces.append(parts)
                self.pending.extend(parts)
            self.folded += 1
            self._cut(last=self.folded == len(self.text_batches))

    def _cut(self, *, last: bool) -> None:
        """Full `/embed` batches from the pending pieces, and the remainder after the last `/tokenize`."""
        while len(self.pending) >= self.size or (last and self.pending):
            self.ready.append((self.embed_batches, self.pending[: self.size]))
            del self.pending[: self.size]
            self.embed_batches += 1

    def result(self) -> tuple[list[list[str]], list[Vector]]:
        return self.pieces, [vector for index in range(self.embed_batches) for vector in self.vectors[index]]


class TeiEncoder:
    """`POST /embed` on a TEI server; one instance per process (`encoder()`)."""

    def __init__(self, *, settings: Settings | None = None, http: requests.Session | None = None):
        self.settings = settings or get_settings()
        self.base = self.settings.tei_url.rstrip("/")
        self.concurrency = self.settings.tei_concurrency
        if http is None:
            http = session(self.settings.tei_api_key)
            # A pooled connection per batch in flight, reused from batch to batch.
            adapter = HTTPAdapter(pool_maxsize=max(DEFAULT_POOLSIZE, self.concurrency))
            http.mount("http://", adapter)
            http.mount("https://", adapter)
        self.http = http
        if self.settings.tei_host:
            # Compose on gateway reaches the cluster's gateway by IP and routes by Host (Known-Issues #7).
            self.http.headers["Host"] = self.settings.tei_host
        if self.settings.tei_lane:
            self.http.headers[LANE_HEADER] = self.settings.tei_lane
        self._info: dict[str, Any] | None = None

    def _send(
        self, method: str, path: str, payload: dict[str, Any] | None = None, tally: _Tally | None = None
    ) -> requests.Response:
        """One request, retried on 429/503 and connection errors with exponential backoff."""
        attempts = self.settings.tei_attempts
        for attempt in range(1, attempts + 1):
            if attempt > 1 and tally is not None:
                tally.retried()
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
                    if tally is not None:
                        tally.served(response.headers.get(LANE_HEADER))
                    return response
                logger.warning("tei-embed answered %s on %s, retrying", response.status_code, path)
            time.sleep(2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    def _call(self, method: str, path: str, payload: dict[str, Any] | None = None, tally: _Tally | None = None) -> Any:
        return self._send(method, path, payload, tally).json()

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

    def _tokenize(self, batch: list[str], tally: _Tally) -> list[list[str]]:
        """Every text of `batch` as the pieces that fit the model's input length (usually one)."""
        limit = self.info["max_input_length"]
        tokenized = self._call("POST", "/tokenize", {"inputs": batch, "add_special_tokens": True}, tally)
        return [split(text, tokens, limit) for text, tokens in zip(batch, tokenized, strict=True)]

    def _embed(self, batch: list[str], tally: _Tally) -> list[Vector]:
        started = time.perf_counter()
        response = self._send("POST", "/embed", {"inputs": batch, "normalize": True, "truncate": True}, tally)
        seconds = time.perf_counter() - started
        metrics.record_embedding("documents", self.name, len(batch), seconds, response.headers.get(LANE_HEADER))
        tally.embedded(len(batch), seconds)
        return response.json()

    def _run(self, texts: Sequence[str], tally: _Tally) -> tuple[list[list[str]], list[Vector]]:
        """Every text's pieces, and every piece's vector in the same order; up to `concurrency` batches in flight."""
        schedule = _Schedule(list(self._batches(texts)), self.batch_size, prefer_embed=self.concurrency > 1)
        calls: dict[str, Callable[[list[str], _Tally], Any]] = {"tokenize": self._tokenize, "embed": self._embed}
        in_flight: dict[Future[Any], tuple[str, int]] = {}
        with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="tei") as pool:
            while True:
                while len(in_flight) < self.concurrency and (request := schedule.next()) is not None:
                    kind, index, batch = request
                    # A copy of this context per request: the span, and `untraced_http()`'s suppression.
                    future = pool.submit(contextvars.copy_context().run, calls[kind], batch, tally)
                    in_flight[future] = (kind, index)
                if not in_flight:
                    return schedule.result()
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    # A batch that failed all its attempts fails the run; those in flight finish first.
                    schedule.answered(*in_flight.pop(future), future.result())

    def documents(self, texts: Sequence[str]) -> list[Vector]:
        """One normalized vector per text; a split text gets the normalized mean of its pieces."""
        if not texts:
            return []
        attributes = {
            "embedding.model": self.name,
            "embedding.texts": len(texts),
            "embedding.batch_size": self.batch_size,
        }
        tally = _Tally(concurrency=self.concurrency)
        with span("embeddings.documents", **attributes), untraced_http():
            try:
                pieces, flat = self._run(texts, tally)
                split_count = sum(1 for parts in pieces if len(parts) > 1)
                if split_count:
                    logger.info("Split %d of %d texts longer than the model's input length", split_count, len(texts))
            finally:
                annotate(**tally.attributes())
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
