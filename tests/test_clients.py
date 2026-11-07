import pytest
import requests

from evilflowers_text.clients import catalog as catalog_module
from evilflowers_text.clients.catalog import CatalogClient, RunNotRunning
from evilflowers_text.clients.http import session
from evilflowers_text.schemas import Box, JobReport, Paragraph, ParagraphLink, Region


class Response:
    def __init__(self, status_code=200, body=b"{}"):
        self.status_code = status_code
        self.content = body
        self.text = body.decode()

    def json(self):
        return {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.posts: list[tuple[str, dict]] = []

    def post(self, url, json, timeout):
        self.posts.append((url, json))
        response = self.responses.pop(0) if self.responses else Response()
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(catalog_module.time, "sleep", lambda _seconds: None)


BOX = Box(x0=0, y0=0, x1=1, y1=1)


def paragraph(seq):
    return Paragraph(page=1, seq=seq, bbox=BOX, text=f"paragraph {seq}")


def test_no_paragraphs_posts_nothing(settings):
    http = Session()
    CatalogClient("run", settings=settings, http=http).paragraphs([], [])
    assert http.posts == []


def test_links_go_with_the_last_batch(settings):
    http = Session()
    client = CatalogClient("run", settings=settings.model_copy(update={"catalog_batch_size": 2}), http=http)
    client.paragraphs([paragraph(seq) for seq in range(5)], [ParagraphLink(source=0, target=1, kind="next")])
    assert [len(body["paragraphs"]) for _url, body in http.posts] == [2, 2, 1]
    assert ["links" in body for _url, body in http.posts] == [False, False, True]
    assert http.posts[0][0] == "http://catalog:8000/extraction/v1/runs/run/paragraphs"


def test_region_payload_has_no_empty_fields(settings):
    http = Session()
    CatalogClient("run", settings=settings, http=http).regions([Region(kind="table", page=1, seq=0, bbox=BOX)])
    [(_url, body)] = http.posts
    assert body == {"regions": [{"kind": "table", "page": 1, "seq": 0, "bbox": BOX.model_dump(), "provenance": {}}]}


def test_server_errors_are_retried(settings):
    http = Session(Response(502), Response(200))
    CatalogClient("run", settings=settings, http=http).job("tables", JobReport(status="done"))
    assert len(http.posts) == 2


def test_connection_errors_give_up_after_the_last_attempt(settings):
    http = Session(requests.ConnectionError("down"), requests.ConnectionError("down"))
    with pytest.raises(requests.ConnectionError):
        CatalogClient("run", settings=settings, http=http).job("tables", JobReport(status="done"))


def test_conflict_means_the_run_is_not_running(settings):
    http = Session(Response(409, b"finished"))
    with pytest.raises(RunNotRunning):
        CatalogClient("run", settings=settings, http=http).job("tables", JobReport(status="done"))
    assert len(http.posts) == 1


def test_job_error_is_truncated():
    assert len(JobReport(status="failed", error="x" * 5000).error) == 2000


def test_session_sends_the_bearer_key_only_when_set():
    from pydantic import SecretStr

    assert session(SecretStr("token")).headers["Authorization"] == "Bearer token"
    assert "Authorization" not in session(SecretStr("")).headers
    assert "Authorization" not in session(None).headers
