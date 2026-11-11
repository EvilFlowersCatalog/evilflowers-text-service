"""Configuration of the worker: one typed object read from the environment (and `.env`).

Every field keeps the environment variable name the deployments already set
(``REDIS_URL``, ``TEXT_QUEUE``, ``CATALOG_API_KEY`` …), so the move from the
two old modules (``settings.py`` and ``config/Config.py``) needs no change in
the k8s manifests.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
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

    # -- Paragraph index (IP-018 Phase 2b) ------------------------------------------------
    #: Paragraphs shorter than this many words are not indexed (tp2025 `MIN_PARAGRAPH_LENGTH`).
    min_paragraph_words: int = 10
    #: `similar` links per paragraph, and the cosine floor below which none is made.
    similar_links_k: int = 5
    similar_min_score: float = 0.6

    def source_path(self, source: str) -> Path:
        """Absolute path of a file the catalog names relative to its storage root."""
        return self.storage_path / source


@lru_cache
def get_settings() -> Settings:
    return Settings()
