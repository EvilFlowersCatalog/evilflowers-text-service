"""Reporting to the catalog's extraction ingestion API (IP-018).

Every evilflowers-*-service posts its results to
`{CATALOG_API_URL}/extraction/v1/runs/{run_id}/...` with the API key of the
catalog's service user. The services never write to the catalog database.
"""

import logging
import time
from typing import Iterable, List, Optional

import requests

import settings

logger = logging.getLogger(__name__)

#: Rows per request; keeps each body well under Django's 2.5 MB upload limit.
BATCH = 100


class RunNotRunning(Exception):
    """The catalog no longer accepts results for this run (409): it was reset or finished."""


class Catalog:
    def __init__(self, run_id: str):
        self.base = f"{settings.CATALOG_API_URL.rstrip('/')}/extraction/v1/runs/{run_id}"
        self.session = requests.Session()
        if settings.CATALOG_API_KEY:
            self.session.headers["Authorization"] = f"Bearer {settings.CATALOG_API_KEY}"

    def _post(self, path: str, payload: dict, attempts: int = 4) -> dict:
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(f"{self.base}{path}", json=payload, timeout=120)
            except requests.RequestException:
                if attempt == attempts:
                    raise
                logger.warning("Catalog unreachable at %s%s, retrying", self.base, path)
            else:
                if response.status_code == 409:
                    raise RunNotRunning(response.text)
                if response.status_code < 500 or attempt == attempts:
                    response.raise_for_status()
                    return response.json() if response.content else {}
                logger.warning("Catalog answered %s on %s, retrying", response.status_code, path)
            time.sleep(2**attempt)
        return {}

    def start(self, page_count: int, page_sizes: List[List[float]], jobs: Iterable[str], provenance: dict) -> None:
        self._post(
            "/start",
            {"page_count": page_count, "page_sizes": page_sizes, "jobs": list(jobs), "provenance": provenance},
        )

    def regions(self, regions: List[dict]) -> None:
        for index in range(0, len(regions), BATCH):
            self._post("/regions", {"regions": regions[index : index + BATCH]})

    def paragraphs(self, paragraphs: List[dict], links: List[dict]) -> None:
        """Paragraphs in batches; the links go with the last batch, once every paragraph exists."""
        if not paragraphs:
            # The catalog requires a non-empty list; a document without a text layer has none.
            return
        batches = [paragraphs[index : index + BATCH] for index in range(0, len(paragraphs), BATCH)]
        for number, batch in enumerate(batches, start=1):
            payload = {"paragraphs": batch}
            if number == len(batches):
                payload["links"] = links
            self._post("/paragraphs", payload)

    def job(self, name: str, *, failed: bool = False, error: Optional[str] = None, provenance: Optional[dict] = None):
        payload = {"status": "failed" if failed else "done", "provenance": provenance or {}}
        if error:
            payload["error"] = error[:2000]
        self._post(f"/jobs/{name}", payload)
