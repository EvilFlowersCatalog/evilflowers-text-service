"""Configuration of the worker: the kit's ``WorkerSettings`` and ``JudgeSettings`` plus text's queues, table review,
TEI and paragraphs.

The broker, delivery semantics (``WORKER_SOFT_SHUTDOWN_TIMEOUT``, ``WORKER_CANCEL_TIMEOUT``, ``MAX_DELIVERIES``),
storage, the catalog ingestion API and logs come from ``evilflowers_worker.settings``. Every field keeps the
environment variable name the deployments already set (``REDIS_URL``, ``TEXT_QUEUE``, ``CATALOG_API_KEY`` …), so the
k8s manifests need no change for it.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import NoDecode

from evilflowers_worker.judges import JudgeSettings
from evilflowers_worker.settings import WorkerSettings

#: The queue both jobs shared before issue #13. Still consumed by default, so messages already
#: in it, or sent by an analyzer that predates the split, are not stranded.
LEGACY_QUEUE = "evilflowers_text_worker"


class Settings(WorkerSettings, JudgeSettings):
    # -- Queues -----------------------------------------------------------------------------------
    #: Where each job is routed (issue #13): tables wait on the table review, paragraphs on the
    #: bulk embed lane, so each gets its own list and can be scaled on its own.
    #: `evilflowers.<service>[.<job>]`, the naming of knowledge#14.
    tables_queue: str = "evilflowers.text.tables"
    paragraphs_queue: str = "evilflowers.text.paragraphs"
    #: The queues this worker consumes, comma-separated (`TEXT_QUEUE`). Empty: both jobs' queues
    #: and the legacy one. One Deployment per job sets just its own.
    queues: Annotated[tuple[str, ...], NoDecode] = Field((), validation_alias=AliasChoices("TEXT_QUEUE", "QUEUE"))

    # -- Scale-in (issue #17) -----------------------------------------------------------------------
    #: Seconds a SIGTERMed worker lets its job run on before it is cancelled and requeued. Short for
    #: text: a book's paragraphs take minutes, and a copy resumes from scratch anyway. Plus
    #: `WORKER_CANCEL_TIMEOUT` (30 s), it must stay under the pod's `terminationGracePeriodSeconds`.
    worker_soft_shutdown_timeout: float = Field(10.0, ge=0)

    # -- Region crops (tables) ------------------------------------------------------------------------
    crop_dpi: int = 144
    webp_quality: int = 82

    # -- Table review: a multimodal model behind an Ollama chat endpoint ---------------------
    #: Checks pdfplumber's markdown against the crop and rewrites it when wrong. Off unless
    #: enabled and an endpoint is set.
    table_review_enabled: bool = False
    #: Full chat URL; in k8s the shared `http://ollama.shared.svc.cluster.local/api/chat`.
    ollama_endpoint: str | None = None
    #: Bearer key, only for Ollama Cloud.
    ollama_api_key: SecretStr | None = None
    #: A model with the `vision` capability.
    table_review_model: str = "qwen3.6:35b-a3b-q4_K_M"
    table_review_timeout: float = 300.0
    #: Longest side of the image sent to the model, in pixels.
    table_review_max_side: int = 1600

    # -- Table decision (issue #20) ---------------------------------------------------------------
    #: "Is this a data table?" goes to decider-vision through the kit's judges (`JudgeSettings`: `DECIDER_ENABLED`,
    #: `DECIDER_VISION_URL`, `DECIDER_VISION_REVISION`, `DECIDER_TIMEOUT`, `JUDGE_THRESHOLDS` with `is_table`, the
    #: budget and breaker). Its `hide` band skips the review. The table score's bands are text's own:
    #: `hide_below,keep_from` on the table score, e.g. `0.4,0.7`. Unset (the default) until calibrated: the region
    #: carries p and score but no band, and the catalog bands it once `EVILFLOWERS_DECIDER_BANDS` holds thresholds.
    decider_table_bands: Annotated[tuple[float, float] | None, NoDecode] = None

    # -- Embeddings: the shared tei-embed (knowledge base, Cookbooks/Shared-Inference) ----------
    #: In k8s the Service; from Compose on gateway `http://192.168.122.227:8081` with `TEI_HOST`.
    #: The served model (multilingual-e5-large-instruct, 1024 dims) is TEI's, read from `/info`;
    #: a different model or revision is a new pipeline version in the catalog.
    tei_url: str = "http://tei-embed.shared.svc.cluster.local"
    #: Bearer key, 1Password `ELVIRA/stu.inference.tei/api_key`.
    tei_api_key: SecretStr | None = None
    #: `Host` header for the gateway route (`embed.elvira.internal`); unset in k8s.
    tei_host: str | None = None
    #: Inputs per request; TEI's `max_client_batch_size` caps it too.
    tei_batch_size: int = 32
    #: Seconds per request (one batch).
    tei_timeout: float = 60.0
    #: The first try plus three retries on 429/503 and connection errors.
    tei_attempts: int = 4
    #: Batches in flight at once within one `documents()` call. 1 keeps the requests strictly
    #: sequential (every `/tokenize`, then every `/embed`); above 1 tokenizing and embedding
    #: overlap, and the router spreads the requests over the bulk lane's replicas (issue #11).
    tei_concurrency: int = Field(1, ge=1)
    #: Sent as `x-embed-lane` on every request when set: `bulk` for this worker, `interactive` is
    #: the catalog's search queries. It beats the router's Host-based choice. Empty: no header.
    tei_lane: Literal["bulk", "interactive"] | None = None

    # -- Paragraph index (IP-018 Phase 2b) ------------------------------------------------
    #: Paragraphs shorter than this many words are not indexed (tp2025 `MIN_PARAGRAPH_LENGTH`).
    min_paragraph_words: int = 10
    #: `similar` links per paragraph, and the cosine floor below which none is made.
    similar_links_k: int = 5
    similar_min_score: float = 0.6
    #: Seconds between the re-reads of the run while a book is embedded (EvilFlowersCatalog#140):
    #: a copy whose attempt was superseded stops within this long instead of embedding the whole book.
    run_recheck_interval: float = Field(60.0, gt=0)

    @field_validator("queues", mode="before")
    @classmethod
    def _split_queues(cls, value: object) -> object:
        """`TEXT_QUEUE=a,b` or a single name, as the manifests have always set it."""
        if isinstance(value, str):
            return tuple(name.strip() for name in value.split(",") if name.strip())
        return value

    @property
    def consumed_queues(self) -> tuple[str, ...]:
        return self.queues or (self.tables_queue, self.paragraphs_queue, LEGACY_QUEUE)

    @field_validator("tei_lane", mode="before")
    @classmethod
    def _empty_lane_is_none(cls, value: object) -> object:
        """`TEI_LANE=` (as `.env.sample` and manifests may leave it) means no lane."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("decider_table_bands", mode="before")
    @classmethod
    def _split_bands(cls, value: object) -> object:
        """`DECIDER_TABLE_BANDS=0.4,0.7`; empty means unset."""
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return tuple(parts) if parts else None
        return value

    @field_validator("decider_table_bands")
    @classmethod
    def _ordered_bands(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and not 0 <= value[0] <= value[1] <= 1:
            raise ValueError("DECIDER_TABLE_BANDS is `hide_below,keep_from` with 0 <= hide_below <= keep_from <= 1")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
