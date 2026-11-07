"""The legacy search-service (Milvus + Elasticsearch), fed by `process_pdf` (IP-018 keeps it, Q6)."""

from typing import Any

import requests

from evilflowers_text.clients.http import session
from evilflowers_text.settings import Settings, get_settings


class SearchServiceClient:
    def __init__(self, *, settings: Settings | None = None, http: requests.Session | None = None):
        self.base = (settings or get_settings()).search_service_url.rstrip("/")
        self.http = http or session()

    def _call(self, method: str, path: str, *, timeout: float, **kwargs: Any) -> Any:
        response = self.http.request(method, f"{self.base}{path}", timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    def index(self, document_id: str, chunks: dict[str, Any]) -> Any:
        return self._call("POST", "/index", json={"document_id": document_id, "chunks": chunks}, timeout=300)

    def delete(self, document_id: str) -> Any:
        return self._call("DELETE", f"/documents/{document_id}", timeout=60)

    def stats(self) -> Any:
        return self._call("GET", "/stats", timeout=30)
