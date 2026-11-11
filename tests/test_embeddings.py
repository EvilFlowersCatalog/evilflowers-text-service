"""The tei-embed client against a fake TEI: batching, long-paragraph splits, retries, model identity."""

import math
import re

import pytest
import requests

from evilflowers_text import embeddings
from evilflowers_text.embeddings import TeiEncoder, split

DIMENSIONS = 4
INFO = {
    "model_id": "intfloat/multilingual-e5-large-instruct",
    "model_sha": "274baa43b0e13e37fafa6428dbc7938e62e5c439",
    "max_input_length": 12,
    "max_client_batch_size": 3,
}


def tokenize(text):
    """One token per word, with TEI's UTF-8 byte offsets and the two special tokens."""
    words = [
        {"text": match.group(), "special": False, "start": len(text[: match.start()].encode()),
         "stop": len(text[: match.end()].encode())}
        for match in re.finditer(r"\S+", text)
    ]  # fmt: skip
    return [{"text": "<s>", "special": True}, *words, {"text": "</s>", "special": True}]


def vector(text):
    """A distinct unit vector per text, so averages are visible."""
    raw = [float(len(text) % 7 + 1), float(text.count("a") + 1), 1.0, float(len(text.split()))]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


class Response:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self.body = body

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeTei:
    """A `requests.Session` stand-in; `failures` are answered (in order) before the real responses."""

    def __init__(self, *failures, info=INFO):
        self.failures = list(failures)
        self.info = info
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, str, dict | None]] = []

    def request(self, method, url, json=None, timeout=None):
        assert timeout == 60.0
        path = url.removeprefix("http://tei")
        self.calls.append((method, path, json))
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return Response(failure)
        if path == "/info":
            return Response(body=self.info)
        assert len(json["inputs"]) <= self.info["max_client_batch_size"]
        if path == "/tokenize":
            return Response(body=[tokenize(text) for text in json["inputs"]])
        assert json["normalize"] is True and json["truncate"] is True
        return Response(body=[vector(text) for text in json["inputs"]])

    def paths(self):
        return [path for _method, path, _json in self.calls]

    def embedded(self):
        return [text for _method, path, body in self.calls if path == "/embed" for text in body["inputs"]]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(embeddings.time, "sleep", lambda _seconds: None)


@pytest.fixture
def tei_settings(settings):
    return settings.model_copy(update={"tei_url": "http://tei/"})


def test_model_identity_is_model_at_revision(tei_settings):
    http = FakeTei()
    encoder = TeiEncoder(settings=tei_settings, http=http)
    assert encoder.name == "intfloat/multilingual-e5-large-instruct@274baa43b0e13e37fafa6428dbc7938e62e5c439"
    assert encoder.name and http.paths() == ["/info"]  # read once


def test_documents_are_plain_text_in_batches_capped_by_tei(tei_settings):
    http = FakeTei()
    texts = [f"paragraph number {index}" for index in range(7)]
    vectors = TeiEncoder(settings=tei_settings, http=http).documents(texts)
    assert vectors == [vector(text) for text in texts]
    # TEI's max_client_batch_size (3) wins over TEI_BATCH_SIZE (32).
    assert [len(body["inputs"]) for _m, path, body in http.calls if path == "/embed"] == [3, 3, 1]
    assert not any(text.startswith("Instruct:") for text in http.embedded())


def test_a_long_paragraph_is_split_at_sentences_and_averaged(tei_settings):
    http = FakeTei()
    first = "Prvá veta má presne päť slov."  # 6 words
    second = "Druhá veta je tiež dosť krátka."
    third = "Tretia veta končí odsek celkom pokojne."
    long = f"{first} {second} {third}"
    [result] = TeiEncoder(settings=tei_settings, http=http).documents([long])
    # 12 tokens per input, 2 of them special: each piece is whole sentences of at most 10 words.
    assert http.embedded() == [first, second, third]
    assert math.isclose(sum(value * value for value in result), 1.0)
    mean = [sum(column) / 3 for column in zip(*(vector(piece) for piece in (first, second, third)), strict=True)]
    norm = math.sqrt(sum(value * value for value in mean))
    assert result == pytest.approx([value / norm for value in mean])


def test_split_cuts_a_sentence_longer_than_the_limit_by_tokens():
    text = " ".join(f"slovo{index}" for index in range(25))
    pieces = split(text, tokenize(text), 12)
    assert [len(piece.split()) for piece in pieces] == [10, 10, 5]
    assert " ".join(pieces) == text


def test_split_maps_byte_offsets_to_characters():
    text = "Žltý kôň úpel ďábelské ódy. " * 3
    pieces = split(text.strip(), tokenize(text.strip()), 7)
    assert pieces == ["Žltý kôň úpel ďábelské ódy."] * 3


def test_short_text_is_not_split():
    assert split("Krátky odsek.", tokenize("Krátky odsek."), 512) == ["Krátky odsek."]


@pytest.mark.parametrize("status", [429, 503])
def test_overload_is_retried(tei_settings, status):
    http = FakeTei(status, requests.ConnectionError("reset"))
    assert TeiEncoder(settings=tei_settings, http=http).info == INFO
    assert http.paths() == ["/info"] * 3


def test_retries_give_up_after_the_last_attempt(tei_settings):
    http = FakeTei(503, 503, 503, 503)
    with pytest.raises(requests.HTTPError):
        TeiEncoder(settings=tei_settings, http=http).documents(["text"])
    assert len(http.calls) == tei_settings.tei_attempts


def test_other_errors_are_not_retried(tei_settings):
    http = FakeTei(413)
    with pytest.raises(requests.HTTPError):
        TeiEncoder(settings=tei_settings, http=http).documents(["text"])
    assert len(http.calls) == 1


def test_host_header_only_when_configured(tei_settings):
    http = FakeTei()
    TeiEncoder(settings=tei_settings, http=http)
    assert "Host" not in http.headers
    TeiEncoder(settings=tei_settings.model_copy(update={"tei_host": "embed.elvira.internal"}), http=http)
    assert http.headers["Host"] == "embed.elvira.internal"


@pytest.fixture
def spans(monkeypatch):
    """Spans `observability.span` creates, finished, through an in-memory exporter."""
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from evilflowers_text.observability import tracing

    exporter = InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing.trace, "get_tracer", lambda *_args, **_kwargs: provider.get_tracer("test"))
    return exporter


def test_one_span_per_call_not_per_batch(tei_settings, spans):
    http = FakeTei()
    encoder = TeiEncoder(settings=tei_settings, http=http)
    encoder.info  # noqa: B018 - read /info first, so the 503 below hits a batch
    http.failures.append(503)
    encoder.documents([f"paragraph number {index}" for index in range(7)])
    [only] = spans.get_finished_spans()
    assert only.name == "embeddings.documents"
    assert only.attributes["embedding.texts"] == 7
    assert only.attributes["embedding.inputs"] == 7
    assert only.attributes["embedding.batches"] == 3
    assert only.attributes["embedding.batch_size"] == 3
    assert only.attributes["tei.retries"] == 1
    assert only.attributes["tei.max_batch_ms"] >= 0
    assert only.attributes["embedding.model"] == encoder.name


def test_a_failed_call_still_reports_its_tally(tei_settings, spans):
    http = FakeTei()
    encoder = TeiEncoder(settings=tei_settings, http=http)
    encoder.info  # noqa: B018
    http.failures.extend([503, 413])
    with pytest.raises(requests.HTTPError):
        encoder.documents(["text"])
    [only] = spans.get_finished_spans()
    assert only.attributes["tei.retries"] == 1
    assert only.attributes["embedding.batches"] == 0
    assert not only.status.is_ok


def test_batch_requests_are_not_traced(tei_settings):
    utils = pytest.importorskip("opentelemetry.instrumentation.utils")
    traced: list[tuple[str, bool]] = []

    class Watching(FakeTei):
        def request(self, method, url, json=None, timeout=None):
            traced.append((url.removeprefix("http://tei"), utils.is_http_instrumentation_enabled()))
            return super().request(method, url, json=json, timeout=timeout)

    TeiEncoder(settings=tei_settings, http=Watching()).documents(["one", "two", "three", "four"])
    assert traced == [
        ("/info", True),
        ("/tokenize", False),
        ("/tokenize", False),
        ("/embed", False),
        ("/embed", False),
    ]
    assert utils.is_http_instrumentation_enabled()


def test_no_texts_no_requests(tei_settings):
    http = FakeTei()
    assert TeiEncoder(settings=tei_settings, http=http).documents([]) == []
    assert http.calls == []
