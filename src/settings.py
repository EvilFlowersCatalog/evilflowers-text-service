"""Settings of the extraction jobs (IP-018); text indexing keeps `config.Config`."""

import os

from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE = os.getenv("TEXT_QUEUE", "evilflowers_text_worker")
#: Request embeddings for the catalog's highlight agent; a separate worker so a
#: reader never waits behind a book being extracted.
EMBED_QUEUE = os.getenv("EMBED_QUEUE", "evilflowers_text_embed")

#: The catalog's storage root (originals are read, crops written).
STORAGE_PATH = os.getenv("STORAGE_PATH", "/mnt/data")

CATALOG_API_URL = os.getenv("CATALOG_API_URL", "http://catalog:8000")
#: API key (bearer JWT) of the catalog user holding `extraction.ingest_extraction`.
CATALOG_API_KEY = os.getenv("CATALOG_API_KEY")
IMAGE_TAG = os.getenv("IMAGE_TAG", "dev")
#: `json` (one object per line, what Loki parses) or `text` (Celery's own format, for a terminal).
LOG_FORMAT = os.getenv("LOG_FORMAT", "json").strip().lower()

CROP_DPI = int(os.getenv("CROP_DPI", "144"))
WEBP_QUALITY = int(os.getenv("WEBP_QUALITY", "82"))

#: Paragraph embeddings (IP-018 Q16). Any sentence-transformers model of the
#: catalog's vector dimension (768): Snowflake Arctic Embed M v2.0 by default,
#: stsb-xlm-r-multilingual selectable. Changing it bumps the catalog's pipeline version.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-m-v2.0")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

#: Paragraphs shorter than this many words are not indexed (tp2025 `MIN_PARAGRAPH_LENGTH`).
MIN_PARAGRAPH_WORDS = int(os.getenv("MIN_PARAGRAPH_WORDS", "10"))
#: `similar` links per paragraph, and the cosine floor below which none is made.
SIMILAR_LINKS_K = int(os.getenv("SIMILAR_LINKS_K", "5"))
SIMILAR_MIN_SCORE = float(os.getenv("SIMILAR_MIN_SCORE", "0.6"))
