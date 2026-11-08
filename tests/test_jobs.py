"""The IP-018 job protocol against a fake catalog: every declared job reports, whatever happens."""

import pytest

from conftest import book
from evilflowers_text.clients.catalog import RunNotRunning
from evilflowers_text.jobs import ParagraphsJob, TablesJob
from evilflowers_text.schemas import EMBEDDING_DIMENSIONS


class FakeCatalog:
    def __init__(self, refuse: bool = False, fail_on: str | None = None):
        self.refuse = refuse
        self.fail_on = fail_on
        self.calls: list[tuple] = []

    def _call(self, name, *payload):
        self.calls.append((name, *payload))
        if self.refuse:
            raise RunNotRunning("finished")
        if name == self.fail_on:
            raise ConnectionError("catalog down")

    def regions(self, regions):
        self._call("regions", regions)

    def paragraphs(self, paragraphs, links):
        self._call("paragraphs", paragraphs, links)

    def job(self, name, report):
        self._call("job", name, report)

    def reports(self):
        return [(call[1], call[2]) for call in self.calls if call[0] == "job"]


class FakeEncoder:
    name = "fake/model@abc123"

    def documents(self, texts):
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1) for _ in texts]


class DownEncoder:
    """tei-embed unreachable: both the embedding and `/info` fail."""

    @property
    def name(self):
        raise ConnectionError("tei-embed down")

    def documents(self, texts):
        raise ConnectionError("tei-embed down")


@pytest.fixture
def stored_book(settings):
    path = settings.storage_path / "catalogs" / "lib" / "book.pdf"
    path.parent.mkdir(parents=True)
    book().save(path)
    return "catalogs/lib/book.pdf"


def paragraphs_job(settings, source, catalog):
    return ParagraphsJob("run-1", source, settings=settings, catalog=catalog, encoder=FakeEncoder())


def test_paragraphs_job_posts_paragraphs_then_reports_done(settings, stored_book):
    catalog = FakeCatalog()
    assert paragraphs_job(settings, stored_book, catalog).run() == "done"
    [(name, rows, links), (_job, _name, report)] = catalog.calls
    assert name == "paragraphs" and len(rows) == 4 and all(len(row.embedding) == EMBEDDING_DIMENSIONS for row in rows)
    assert {link.kind for link in links} == {"next", "similar"}
    assert report.status == "done"
    # model@revision from TEI's /info: the catalog never compares vectors of two models.
    assert report.provenance == {"image": "test", "embedding_model": "fake/model@abc123"}


def test_unreachable_tei_is_still_reported_as_failed(settings, stored_book):
    catalog = FakeCatalog()
    job = ParagraphsJob("run-1", stored_book, settings=settings, catalog=catalog, encoder=DownEncoder())
    assert job.run() == "failed"
    [(name, report)] = catalog.reports()
    assert name == "paragraphs" and report.status == "failed" and "tei-embed down" in report.error
    assert report.provenance == {"image": "test"}


def test_unreadable_pdf_reports_failed(settings):
    (settings.storage_path / "broken.pdf").write_bytes(b"not a pdf")
    catalog = FakeCatalog()
    assert paragraphs_job(settings, "broken.pdf", catalog).run() == "failed"
    [(name, report)] = catalog.reports()
    assert name == "paragraphs" and report.status == "failed" and report.error


def test_failed_publish_is_still_reported(settings, stored_book):
    catalog = FakeCatalog(fail_on="paragraphs")
    assert paragraphs_job(settings, stored_book, catalog).run() == "failed"
    [(_name, report)] = catalog.reports()
    assert report.status == "failed" and "catalog down" in report.error


def test_a_run_that_is_not_running_is_left_alone(settings, stored_book):
    catalog = FakeCatalog(refuse=True)
    assert paragraphs_job(settings, stored_book, catalog).run() == "not_running"
    assert catalog.reports() == []


def test_tables_job_reports_done_even_without_regions(settings, stored_book):
    catalog = FakeCatalog()
    assert TablesJob("run-1", stored_book, "runs/1", [], settings=settings, catalog=catalog).run() == "done"
    [(_name, report)] = catalog.reports()
    assert report.provenance["extractor"] == "pdfplumber" and report.provenance["image"] == "test"


def test_tables_job_rejects_a_malformed_region_as_a_failed_job(settings, stored_book):
    catalog = FakeCatalog()
    job = TablesJob(
        "run-1", stored_book, "runs/1", [{"page": 0, "seq": 1, "bbox": {}}], settings=settings, catalog=catalog
    )
    assert job.run() == "failed"
    [(name, report)] = catalog.reports()
    assert name == "tables" and "ValidationError" in report.error
