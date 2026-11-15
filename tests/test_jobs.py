"""The two leaf jobs on the kit's protocol: every declared job reports, under its own name, for its own attempt."""

from urllib.parse import parse_qs, urlsplit

import pytest

from conftest import book
from evilflowers_text.jobs import ParagraphsJob, TablesJob
from evilflowers_worker import contract
from evilflowers_worker.catalog import CatalogClient
from evilflowers_worker.contract import EMBEDDING_DIMENSIONS, RegionsJob
from evilflowers_worker.testing import FakeCatalog, StubCatalogServer, running

PENDING = {"paragraphs": "pending", "tables": "pending", "figures": "done"}
SOURCE = "catalogs/lib/book.pdf"
OUTPUT = "catalogs/lib/extraction/a/v2"


class FakeEncoder:
    name = "fake/model@abc123"

    def __init__(self):
        self.texts = 0

    def documents(self, texts, *, between=None):
        vectors = []
        for _text in texts:
            self.texts += 1
            vectors.append([1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1))
            if between is not None:
                between()
        return vectors


class DownEncoder:
    """tei-embed unreachable: both the embedding and `/info` fail."""

    @property
    def name(self):
        raise ConnectionError("tei-embed down")

    def documents(self, texts, *, between=None):
        raise ConnectionError("tei-embed down")


@pytest.fixture
def stored_book(settings):
    path = settings.storage_path / SOURCE
    path.parent.mkdir(parents=True)
    book().save(path)
    return SOURCE


def paragraphs_message(**kwargs) -> contract.ParagraphsJob:
    return contract.ParagraphsJob.model_validate({"run_id": "run-1", "source": SOURCE, "output_dir": OUTPUT, **kwargs})


def tables_message(**kwargs) -> RegionsJob:
    return RegionsJob.model_validate({"run_id": "run-1", "source": SOURCE, "output_dir": OUTPUT, **kwargs})


def paragraphs_job(settings, catalog, encoder=None, **kwargs) -> ParagraphsJob:
    return ParagraphsJob(
        paragraphs_message(**kwargs), settings=settings, catalog=catalog, encoder=encoder or FakeEncoder()
    )


def catalog_with(jobs=None, **kwargs) -> FakeCatalog:
    return FakeCatalog(states=[running(jobs=PENDING if jobs is None else jobs, **kwargs)])


# -- paragraphs -----------------------------------------------------------------------------------------------------


def test_paragraphs_job_posts_paragraphs_then_reports_done(settings, stored_book):
    catalog = catalog_with()
    assert paragraphs_job(settings, catalog).run() == "done"
    assert len(catalog.paragraphs_posted) == 4
    assert all(len(row["embedding"]) == EMBEDDING_DIMENSIONS for row in catalog.paragraphs_posted)
    assert {link["kind"] for link in catalog.links_posted} == {"next", "similar"}
    [report] = catalog.jobs
    assert report["name"] == "paragraphs" and report["status"] == "done"
    # model@revision from TEI's /info: the catalog never compares vectors of two models.
    assert report["provenance"] == {"image": "test", "outline": "bookmarks", "embedding_model": "fake/model@abc123"}


def test_without_an_outline_the_sections_are_the_bookmarks(settings, stored_book):
    catalog = catalog_with()
    paragraphs_job(settings, catalog).run()
    assert [row["section"] for row in catalog.paragraphs_posted] == ["Information", "Information", "Coding", "Coding"]


def test_the_analyzers_outline_replaces_the_bookmarks(settings, stored_book):
    """Issue #15: books without bookmarks get the headings the analyzer detected."""
    catalog = catalog_with()
    outline = [[1, "Prologue", 1], [2, "Detected heading", 2]]
    assert paragraphs_job(settings, catalog, outline=outline).run() == "done"
    sections = [row["section"] for row in catalog.paragraphs_posted]
    assert sections == ["Prologue", "Detected heading", "Detected heading", "Detected heading"]
    assert catalog.jobs[0]["provenance"]["outline"] == "analyzer"


def test_an_empty_outline_is_authoritative(settings, stored_book):
    """`[]`: the analyzer found no outline; the PDF's bookmarks must not come back in."""
    catalog = catalog_with()
    paragraphs_job(settings, catalog, outline=[]).run()
    assert all("section" not in row for row in catalog.paragraphs_posted)


def test_control_characters_are_stripped_from_section_titles(settings, stored_book):
    """Issue #18: a UTF-16 title without its BOM carries NULs, and the catalog refuses NUL in `section` (422)."""
    seen = []

    class Recording(FakeEncoder):
        def documents(self, texts, *, between=None):
            seen.extend(texts)
            return super().documents(texts)

    catalog = catalog_with()
    outline = [[1, "Prologue", 1], [1, "\x00I\x00n\x00t\x00r\x00o\x7f\t", 2]]
    assert paragraphs_job(settings, catalog, encoder=Recording(), outline=outline).run() == "done"
    assert [row["section"] for row in catalog.paragraphs_posted] == ["Prologue", "Intro", "Intro", "Intro"]
    assert not any("\x00" in text for text in seen)


def test_a_title_of_only_control_characters_is_skipped(settings, stored_book):
    catalog = catalog_with()
    outline = [[1, "Prologue", 1], [1, "\x00\x00\x1f", 2]]
    paragraphs_job(settings, catalog, outline=outline).run()
    assert [row["section"] for row in catalog.paragraphs_posted] == ["Prologue"] * 4


def test_section_prefixes_the_embedded_text(settings, stored_book):
    seen = []

    class Recording(FakeEncoder):
        def documents(self, texts, *, between=None):
            seen.extend(texts)
            return super().documents(texts)

    paragraphs_job(settings, catalog_with(), encoder=Recording()).run()
    assert seen[0].startswith("Information\nEntropy measures")


def test_unreachable_tei_is_still_reported_as_failed(settings, stored_book):
    catalog = catalog_with()
    assert paragraphs_job(settings, catalog, encoder=DownEncoder()).run() == "failed"
    [report] = catalog.jobs
    assert report["name"] == "paragraphs" and report["failed"] and "tei-embed down" in report["error"]


def test_unreadable_pdf_reports_failed(settings):
    (settings.storage_path / "catalogs/lib").mkdir(parents=True)
    (settings.storage_path / SOURCE).write_bytes(b"not a pdf")
    catalog = catalog_with()
    assert paragraphs_job(settings, catalog).run() == "failed"
    [report] = catalog.jobs
    assert report["name"] == "paragraphs" and report["failed"] and report["error"]


def test_failed_publish_is_still_reported(settings, stored_book):
    catalog = catalog_with()
    catalog.fail_on["paragraphs"] = ConnectionError("catalog down")
    assert paragraphs_job(settings, catalog).run() == "failed"
    [report] = catalog.jobs
    assert report["failed"] and "catalog down" in report["error"]


@pytest.mark.parametrize(
    ("status", "jobs", "attempt", "outcome"),
    [
        # The case of issue #12: paragraphs stored, the run waits for tables and equations.
        ("running", {"paragraphs": "done", "tables": "pending"}, None, "already_done"),
        ("running", {"paragraphs": "failed", "tables": "pending"}, None, "already_done"),
        ("running", {"tables": "pending"}, None, "already_done"),
        ("done", {"paragraphs": "done"}, None, "not_running"),
        ("failed", {"paragraphs": "pending"}, None, "not_running"),
        # Issue #14: a copy from an earlier attempt, its job `pending` again after the new `/start`.
        ("running", {"paragraphs": "pending"}, 4, "superseded"),
    ],
)
def test_a_copy_of_a_job_stops_before_the_work(settings, stored_book, status, jobs, attempt, outcome):
    catalog = FakeCatalog(states=[running(status=status, jobs=jobs, attempt=attempt)])
    encoder = FakeEncoder()
    assert paragraphs_job(settings, catalog, encoder, attempt=3).run() == outcome
    # Nothing embedded or posted.
    assert encoder.texts == 0
    assert catalog.paragraphs_posted == []
    # Only a superseded copy says so (the catalog counts them); the others stay silent.
    assert [report["status"] for report in catalog.jobs] == (["superseded"] if outcome == "superseded" else [])


def test_a_superseded_copy_stops_between_embedding_batches(settings, stored_book):
    """A new `/start` while the book is embedded: the copy stops before posting anything."""
    catalog = FakeCatalog(
        states=[running(jobs=PENDING, attempt=3), running(jobs=PENDING, attempt=4)],
    )
    clock = iter(range(0, 10_000, 1000))
    encoder = FakeEncoder()
    job = ParagraphsJob(
        paragraphs_message(attempt=3),
        settings=settings,
        catalog=catalog,
        encoder=encoder,
        recheck_interval=60,
        clock=lambda: next(clock),
    )
    assert job.run() == "superseded"
    assert encoder.texts == 1  # stopped after the first batch
    assert catalog.paragraphs_posted == []
    assert [report["status"] for report in catalog.jobs] == ["superseded"]


def test_report_failure_reports_without_running(settings, stored_book):
    catalog = catalog_with()
    encoder = FakeEncoder()
    job = paragraphs_job(settings, catalog, encoder)
    assert job.report_failure(RuntimeError("Worker exited prematurely: signal 9 (SIGKILL)")) == "failed"
    [report] = catalog.jobs
    assert report["name"] == "paragraphs" and report["failed"] and "SIGKILL" in report["error"]
    assert encoder.texts == 0


def test_a_job_past_the_delivery_cap_is_given_up(settings, stored_book):
    catalog = catalog_with()
    encoder = FakeEncoder()
    job = ParagraphsJob(
        paragraphs_message(), settings=settings, catalog=catalog, encoder=encoder, delivery=settings.max_deliveries + 1
    )
    assert job.run() == "failed"
    assert encoder.texts == 0 and catalog.reads == 0
    [report] = catalog.jobs
    assert report["failed"] and "Gave up" in report["error"]


# -- tables ---------------------------------------------------------------------------------------------------------


def test_tables_job_reports_done_even_without_regions(settings, stored_book):
    catalog = catalog_with()
    assert TablesJob(tables_message(), settings=settings, catalog=catalog).run() == "done"
    [report] = catalog.jobs
    assert report["name"] == "tables"
    assert report["provenance"]["extractor"] == "pdfplumber" and report["provenance"]["image"] == "test"


def test_a_batch_reports_under_its_own_job(settings, stored_book):
    """Per-batch dispatch (#14): the batch's job, not `tables`, which the run no longer declares."""
    catalog = catalog_with(jobs={"tables:0001-0008": "pending", "tables:0009-0016": "pending"})
    job = TablesJob(tables_message(job="tables:0001-0008"), settings=settings, catalog=catalog)
    assert job.run() == "done"
    assert [report["name"] for report in catalog.jobs] == ["tables:0001-0008"]


def test_a_reported_batch_is_not_redone(settings, stored_book):
    catalog = catalog_with(jobs={"tables:0001-0008": "done", "tables": "pending"})
    job = TablesJob(tables_message(job="tables:0001-0008"), settings=settings, catalog=catalog)
    assert job.run() == "already_done"
    assert catalog.jobs == []


def test_tables_copy_stops_before_the_work(settings, stored_book):
    catalog = catalog_with(jobs={"paragraphs": "pending", "tables": "done"})
    box = {"x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.5}
    job = TablesJob(tables_message(regions=[{"page": 9, "seq": 0, "bbox": box}]), settings=settings, catalog=catalog)
    assert job.run() == "already_done"
    assert catalog.regions_posted == [] and catalog.jobs == []


def test_an_output_dir_outside_the_storage_fails_the_job(settings, stored_book):
    catalog = catalog_with()
    message = RegionsJob.model_validate({"run_id": "run-1", "source": SOURCE, "output_dir": "../../etc"})
    assert TablesJob(message, settings=settings, catalog=catalog).run() == "failed"
    [report] = catalog.jobs
    assert report["failed"] and "escapes STORAGE_PATH" in report["error"]


# -- the wire: ?attempt= and the job's path, on a real client ----------------------------------------------------


def test_every_write_of_both_jobs_carries_the_attempt(settings, stored_book):
    with StubCatalogServer(attempt=3, run={"status": "running", "jobs": PENDING, "attempt": 3}) as stub:
        live = settings.model_copy(update={"catalog_api_url": stub.url})
        for job in (
            ParagraphsJob(
                paragraphs_message(attempt=3),
                settings=live,
                catalog=CatalogClient("run-1", settings=live, attempt=3),
                encoder=FakeEncoder(),
            ),
            TablesJob(tables_message(attempt=3), settings=live),
        ):
            assert job.run() == "done"

    writes = [(path, body) for method, path, body in stub.calls if method == "POST"]
    paths = [urlsplit(path).path for path, _body in writes]
    assert paths == [
        "/extraction/v1/runs/run-1/paragraphs",
        "/extraction/v1/runs/run-1/jobs/paragraphs",
        "/extraction/v1/runs/run-1/jobs/tables",
    ]
    assert all(parse_qs(urlsplit(path).query) == {"attempt": ["3"]} for path, _body in writes)


def test_a_batch_reports_to_its_own_path(settings, stored_book):
    run = {"status": "running", "jobs": {"tables:0001-0008": "pending"}, "attempt": 3}
    with StubCatalogServer(attempt=3, run=run) as stub:
        live = settings.model_copy(update={"catalog_api_url": stub.url})
        assert TablesJob(tables_message(attempt=3, job="tables:0001-0008"), settings=live).run() == "done"
    [(_method, path, body)] = [call for call in stub.calls if call[0] == "POST"]
    assert path == "/extraction/v1/runs/run-1/jobs/tables:0001-0008?attempt=3"
    assert body["status"] == "done"
