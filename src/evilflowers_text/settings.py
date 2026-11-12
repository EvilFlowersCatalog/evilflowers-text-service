"""Configuration of the worker: one typed object read from the environment (and `.env`).

Every field keeps the environment variable name the deployments already set
(``REDIS_URL``, ``TEXT_QUEUE``, ``CATALOG_API_KEY`` …), so the move from the
two old modules (``settings.py`` and ``config/Config.py``) needs no change in
the k8s manifests.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    # -- Celery -------------------------------------------------------------------
    redis_url: str = "redis://redis:6379/0"
    #: The extraction jobs.
    queue: str = Field("evilflowers_text_worker", validation_alias=AliasChoices("TEXT_QUEUE", "QUEUE"))
    #: Seconds before Redis redelivers an unacknowledged message. Every consumer of a Redis db
    #: restores *all* unacked messages older than its own timeout, in any queue, so this must
    #: outlast the longest task of any service sharing the db, not only this one's
    #: (an equation pass took over 2 h; EvilFlowersCatalog#120).
    broker_visibility_timeout: int = 43200

    # -- Storage and provenance -----------------------------------------------------
    #: The catalog's storage root: originals are read here, crops written.
    storage_path: Path = Path("/mnt/data")
    #: Built into the image by CI; recorded in every run's provenance and as `service.version`.
    image_tag: str = "dev"

    # -- Catalog ingestion API (IP-018) ---------------------------------------------
    catalog_api_url: str = "http://catalog:8000"
    #: API key (bearer JWT) of the catalog user holding `extraction.ingest_extraction`.
    catalog_api_key: SecretStr | None = None
    #: Rows per request; keeps each body well under Django's 2.5 MB upload limit.
    catalog_batch_size: int = 100
    catalog_attempts: int = 4
    catalog_timeout: float = 120.0

    # -- Logs ---------------------------------------------------------------------------
    #: `json` (one object per line, what Loki parses) or `text` (Celery's own format, for a terminal).
    log_format: Literal["json", "text"] = "json"

    # -- Region crops (tables) ------------------------------------------------------------
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

    @field_validator("tei_lane", mode="before")
    @classmethod
    def _empty_lane_is_none(cls, value: object) -> object:
        """`TEI_LANE=` (as `.env.sample` and manifests may leave it) means no lane."""
        return None if isinstance(value, str) and not value.strip() else value

    def source_path(self, source: str) -> Path:
        """Absolute path of a file the catalog names relative to its storage root."""
        return self.storage_path / source


@lru_cache
def get_settings() -> Settings:
    return Settings()
