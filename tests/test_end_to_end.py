"""One ``tables`` batch through the production text worker, a real Redis and the catalog's API on localhost.

What only this test sees: the task wrapper (``run_job``: validation, delivery count), the kit's app wiring in a real
prefork worker, and the requests on the wire (the batch's own ``/jobs/<name>`` and ``?attempt=``).
Needs ``TEST_REDIS_URL`` (default ``redis://localhost:6379/15``); skipped without one, a failure in CI.
"""

import os

import pytest

from conftest import book
from evilflowers_worker.testing import StubCatalogServer, WorkerProcess, redis_client, wait_for

REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
SOURCE = "catalogs/c/e/book.pdf"
OUTPUT = "catalogs/c/e/extraction/a/v2"


@pytest.fixture
def broker():
    if redis_client(REDIS_URL) is None:
        if os.environ.get("CI"):
            pytest.fail(f"no Redis at {REDIS_URL}: the real-broker tests must run in CI")
        pytest.skip(f"no Redis at {REDIS_URL}")
    return REDIS_URL


def test_a_tables_batch_reports_under_its_own_job_with_its_attempt(broker, tmp_path):
    path = tmp_path / SOURCE
    path.parent.mkdir(parents=True)
    book().save(path)
    run = {"status": "running", "jobs": {"tables:0001-0008": "pending"}, "attempt": 3}
    box = {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.5}

    with StubCatalogServer(attempt=3, run=run) as catalog:
        env = {"CATALOG_API_URL": catalog.url, "STORAGE_PATH": str(tmp_path), "IMAGE_TAG": "e2e"}
        with WorkerProcess("evilflowers_text.worker", redis_url=broker, env=env) as worker:
            worker.wait_ready()
            # The test db is shared: other suites may have left unacknowledged messages behind.
            baseline = worker.unacked()
            worker.send(
                "evilflowers_text_worker.tables",
                run_id="run-1",
                source=SOURCE,
                output_dir=OUTPUT,
                attempt=3,
                job="tables:0001-0008",
                regions=[{"page": 1, "seq": 0, "bbox": box, "caption": "Table 1"}],
                added_by_a_newer_analyzer=True,
            )
            reported = "/extraction/v1/runs/run-1/jobs/tables:0001-0008?attempt=3"
            assert wait_for(lambda: ("POST", reported) in catalog.paths(), 60), worker.output
            # Acknowledged once reported: nothing left for Redis to restore.
            assert wait_for(lambda: worker.unacked() == baseline and worker.redis.llen(worker.queue) == 0, 10)

    assert catalog.paths() == [
        ("GET", "/extraction/v1/runs/run-1"),
        ("POST", "/extraction/v1/runs/run-1/regions?attempt=3"),
        ("POST", reported),
    ]
    _read, (_, _, regions), (_, _, report) = catalog.calls
    assert [region["image_path"] for region in regions["regions"]] == [f"{OUTPUT}/p001-tab00.webp"]
    assert (tmp_path / OUTPUT / "p001-tab00.webp").stat().st_size > 0
    assert report["status"] == "done" and report["provenance"]["image"] == "e2e"


def test_an_invalid_message_is_reported_failed_and_not_retried(broker, tmp_path):
    run = {"status": "running", "jobs": {"paragraphs": "pending"}, "attempt": 3}
    with StubCatalogServer(attempt=3, run=run) as catalog:
        env = {"CATALOG_API_URL": catalog.url, "STORAGE_PATH": str(tmp_path)}
        with WorkerProcess("evilflowers_text.worker", redis_url=broker, env=env) as worker:
            worker.wait_ready()
            baseline = worker.unacked()
            # No `source`: the message breaks the contract.
            worker.send("evilflowers_text_worker.paragraphs", run_id="run-1", output_dir=OUTPUT, attempt=3)
            reported = "/extraction/v1/runs/run-1/jobs/paragraphs?attempt=3"
            assert wait_for(lambda: ("POST", reported) in catalog.paths(), 60), worker.output
            assert wait_for(lambda: worker.unacked() == baseline and worker.redis.llen(worker.queue) == 0, 10)

    [(_method, _path, report)] = catalog.calls
    assert report["status"] == "failed" and "Invalid message" in report["error"]
