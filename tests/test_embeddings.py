"""The tei-embed client against a fake TEI: batching, splits, retries, model identity, concurrency, lanes."""

import json
import math
import re
import threading

import pytest
import requests
import requests.adapters

from evilflowers_text import embeddings
from evilflowers_text.embeddings import TeiEncoder, split
from evilflowers_text.settings import Settings

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
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeTei:
    """A `requests.Session` stand-in; `failures` are answered (in order) before the real responses."""

    def __init__(self, *failures, info=INFO, lane=None):
        self.failures = list(failures)
        self.info = info
        #: The lane the fake router says served each request (`x-embed-lane` on the response).
        self.lane = lane
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, str, dict | None]] = []
        self._lock = threading.Lock()

    def request(self, method, url, json=None, timeout=None):
        assert timeout == 60.0
        path = url.removeprefix("http://tei")
        with self._lock:
            self.calls.append((method, path, json))
            failure = self.failures.pop(0) if self.failures else None
        if isinstance(failure, Exception):
            raise failure
        if failure is not None:
            return Response(failure)
        return self.answer(path, json)

    def answer(self, path, json):
        headers = {"x-embed-lane": self.lane} if self.lane else {}
        if path == "/info":
            return Response(body=self.info, headers=headers)
        assert len(json["inputs"]) <= self.info["max_client_batch_size"]
        if path == "/tokenize":
            return Response(body=[tokenize(text) for text in json["inputs"]], headers=headers)
        assert json["normalize"] is True and json["truncate"] is True
        return Response(body=[vector(text) for text in json["inputs"]], headers=headers)

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


# -- Concurrency and lanes (issue #11) ---------------------------------------------------------------

LONG = "Prvá veta má presne päť slov. Druhá veta je tiež dosť krátka. Tretia veta končí odsek celkom pokojne."
#: 20 texts, one of them split in three: 7 /tokenize batches and 8 /embed batches of at most 3.
TEXTS = [f"paragraph number {index}" for index in range(10)] + [LONG] + [f"odsek číslo {index}" for index in range(9)]


def with_concurrency(settings, concurrency, **update):
    return settings.model_copy(update={"tei_concurrency": concurrency, **update})


def embed_batches(http):
    return [tuple(body["inputs"]) for _method, path, body in http.calls if path == "/embed"]


class OutOfOrderTei(FakeTei):
    """Holds the first `/embed` until a later one has been answered, and counts the requests in flight."""

    def __init__(self, *failures, **kwargs):
        super().__init__(*failures, **kwargs)
        self.active = self.peak = 0
        self.held = False
        self.overtaken = threading.Event()

    def request(self, method, url, json=None, timeout=None):
        path = url.removeprefix("http://tei")
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            hold = path == "/embed" and not self.held
            self.held = self.held or hold
        try:
            if hold:
                assert self.overtaken.wait(timeout=5), "no later /embed was answered while the first was held"
            return super().request(method, url, json=json, timeout=timeout)
        finally:
            with self._lock:
                self.active -= 1
            if path == "/embed" and not hold:
                self.overtaken.set()


def test_concurrency_one_keeps_the_sequential_requests(tei_settings):
    http = FakeTei()
    vectors = TeiEncoder(settings=with_concurrency(tei_settings, 1), http=http).documents(TEXTS)
    # Every /tokenize, then every /embed, the pieces cut into batches in input order: as before issue #11.
    assert http.paths() == ["/info"] + ["/tokenize"] * 7 + ["/embed"] * 8
    pieces = [piece for text in TEXTS for piece in (split(text, tokenize(text), 12))]
    assert embed_batches(http) == [tuple(pieces[start : start + 3]) for start in range(0, len(pieces), 3)]
    assert len(vectors) == len(TEXTS)


def test_concurrent_batches_come_back_in_input_order(tei_settings):
    sequential = FakeTei()
    expected = TeiEncoder(settings=with_concurrency(tei_settings, 1), http=sequential).documents(TEXTS)
    http = OutOfOrderTei()
    vectors = TeiEncoder(settings=with_concurrency(tei_settings, 3), http=http).documents(TEXTS)
    assert http.overtaken.is_set()  # TEI answered out of order
    assert 2 <= http.peak <= 3
    assert vectors == expected  # same order, and the split text still averaged from its own pieces
    assert sorted(embed_batches(http)) == sorted(embed_batches(sequential))  # the same /embed requests


def test_concurrent_requests_pipeline_tokenize_and_embed(tei_settings):
    http = FakeTei()
    TeiEncoder(settings=with_concurrency(tei_settings, 3), http=http).documents(TEXTS)
    paths = http.paths()
    # An /embed went out before the last /tokenize: the run is not tokenized in full first.
    assert paths.index("/embed") < len(paths) - 1 - paths[::-1].index("/tokenize")


def test_concurrent_retry_repeats_only_that_batch(tei_settings):
    http = FakeTei()
    encoder = TeiEncoder(settings=with_concurrency(tei_settings, 3), http=http)
    encoder.info  # noqa: B018 - the 503 below hits a batch
    http.failures.append(503)
    vectors = encoder.documents(TEXTS)
    assert len(http.calls) == 1 + 7 + 8 + 1
    assert vectors == TeiEncoder(settings=tei_settings, http=FakeTei()).documents(TEXTS)


def test_concurrent_run_fails_after_the_attempts_of_one_batch(tei_settings):
    class Poisoned(FakeTei):
        def answer(self, path, json):
            if path == "/embed" and "paragraph number 4" in json["inputs"]:
                return Response(503)
            return super().answer(path, json)

    http = Poisoned()
    with pytest.raises(requests.HTTPError):
        TeiEncoder(settings=with_concurrency(tei_settings, 3), http=http).documents(TEXTS)
    poisoned = [batch for batch in embed_batches(http) if "paragraph number 4" in batch]
    assert len(poisoned) == tei_settings.tei_attempts
    assert len(set(poisoned)) == 1


class Recording(requests.adapters.HTTPAdapter):
    """A real `requests.Session` transport answering like `FakeTei`, keeping every request sent."""

    def __init__(self):
        super().__init__()
        self.sent: list[requests.PreparedRequest] = []
        self.fake = FakeTei(lane="bulk")

    def send(self, request, **kwargs):
        self.sent.append(request)
        path = request.path_url
        answer = self.fake.answer(path, json.loads(request.body) if request.body else None)
        response = requests.Response()
        response.status_code = answer.status_code
        response.headers.update(answer.headers)
        response._content = json.dumps(answer.body).encode()
        response.request = request
        response.url = request.url
        return response


@pytest.mark.parametrize(("lane", "header"), [("bulk", "bulk"), (None, None), ("", None)])
def test_lane_header_on_every_request_only_when_configured(tei_settings, lane, header):
    transport = Recording()
    http = requests.Session()
    http.mount("http://", transport)
    # Through the validator, as `TEI_LANE=` in the environment would be.
    settings = with_concurrency(tei_settings, 2, tei_lane=Settings(_env_file=None, tei_lane=lane).tei_lane)
    TeiEncoder(settings=settings, http=http).documents(["one", "two", "three", "four"])
    assert {request.path_url for request in transport.sent} == {"/info", "/tokenize", "/embed"}
    assert [request.headers.get("x-embed-lane") for request in transport.sent] == [header] * len(transport.sent)


def test_span_and_metric_carry_concurrency_and_serving_lane(tei_settings, spans, monkeypatch):
    recorded = []
    monkeypatch.setattr(embeddings.metrics, "record_embedding", lambda *args: recorded.append(args))
    http = FakeTei(lane="bulk")
    TeiEncoder(settings=with_concurrency(tei_settings, 2, tei_lane="bulk"), http=http).documents(TEXTS)
    [only] = spans.get_finished_spans()
    assert only.attributes["tei.concurrency"] == 2
    assert only.attributes["tei.lane"] == "bulk"
    assert {args[-1] for args in recorded} == {"bulk"}
    assert len(recorded) == 8


@pytest.mark.parametrize(("concurrency", "pool"), [(1, 10), (16, 16)])
def test_own_session_pools_a_connection_per_batch_in_flight(tei_settings, concurrency, pool):
    encoder = TeiEncoder(settings=with_concurrency(tei_settings, concurrency))
    for scheme in ("http://", "https://"):
        assert encoder.http.get_adapter(f"{scheme}tei/")._pool_maxsize == pool


def test_no_lane_attribute_when_the_router_does_not_say(tei_settings, spans):
    TeiEncoder(settings=tei_settings, http=FakeTei()).documents(["text"])
    [only] = spans.get_finished_spans()
    assert "tei.lane" not in only.attributes
    assert only.attributes["tei.concurrency"] == 1
