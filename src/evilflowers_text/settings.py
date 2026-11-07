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
    #: Extraction and indexing jobs.
    queue: str = Field("evilflowers_text_worker", validation_alias=AliasChoices("TEXT_QUEUE", "QUEUE"))
    #: Request embeddings for the catalog's reader; a separate worker so a reader
    #: never waits behind a book being extracted.
    embed_queue: str = "evilflowers_text_embed"
    #: Seconds before Redis redelivers an unacknowledged message. The extraction jobs
    #: acknowledge late, so this must outlast the longest book (about 30 minutes).
    broker_visibility_timeout: int = 7200
    #: Seconds a task result (an `embed_query` vector) stays in Redis.
    result_expires: int = 3600

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

    # -- Paragraph index (IP-018 Phase 2b) ------------------------------------------------
    #: Any sentence-transformers model of the catalog's vector dimension (768). Changing it
    #: bumps the catalog's pipeline version.
    embedding_model: str = "Snowflake/snowflake-arctic-embed-m-v2.0"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    #: Paragraphs shorter than this many words are not indexed (tp2025 `MIN_PARAGRAPH_LENGTH`).
    min_paragraph_words: int = 10
    #: `similar` links per paragraph, and the cosine floor below which none is made.
    similar_links_k: int = 5
    similar_min_score: float = 0.6

    # -- Search-service indexing (`process_pdf`) ----------------------------------------------
    search_service_url: str = "http://localhost:8001"
    chunk_size: int = 768
    chunk_overlap: int = 50
    #: Table extractor of the indexing pipeline: `PdfPlumber` or `Camelot`.
    table_extractor: str = "PdfPlumber"
    table_extractor_mode: str = "lattice"
    #: DocLayout-YOLO weights for ToC detection; fetched into the Hugging Face cache when absent.
    yolo_model_path: Path | None = None
    yolo_repo_id: str = "juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained"
    yolo_filename: str = "doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"

    def source_path(self, source: str) -> Path:
        """Absolute path of a file the catalog names relative to its storage root."""
        return self.storage_path / source


@lru_cache
def get_settings() -> Settings:
    return Settings()
