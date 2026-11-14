"""The task contract with the analyzer: names, queues and argument shapes; and shutdown."""

import inspect
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from billiard.exceptions import WorkerLostError
from celery import Celery
from celery.concurrency.solo import TaskPool as SoloPool

from evilflowers_text import worker as worker_module
from evilflowers_text.settings import Settings
from evilflowers_text.tasks import extraction
from evilflowers_text.worker import CeleryConfig, app, requeue_on_sigterm

TASKS = {
    "evilflowers_text_worker.tables": "evilflowers.text.tables",
    "evilflowers_text_worker.paragraphs": "evilflowers.text.paragraphs",
}


def test_every_task_is_registered_and_routed_to_its_own_queue():
    app.loader.import_default_modules()
    for name, queue in TASKS.items():
        assert name in app.tasks, name
        assert app.amqp.router.route({}, name)["queue"].name == queue


def test_the_worker_consumes_both_queues_and_the_legacy_one_by_default(monkeypatch):
    monkeypatch.delenv("TEXT_QUEUE", raising=False)
    config = CeleryConfig(Settings(_env_file=None))
    assert [queue.name for queue in config.task_queues] == [
        "evilflowers.text.tables",
        "evilflowers.text.paragraphs",
        "evilflowers_text_worker",
    ]


@pytest.mark.parametrize(
    ("value", "queues"),
    [
        # As the manifests set it before issue #13.
        ("evilflowers_text_worker", ("evilflowers_text_worker",)),
        ("evilflowers.text.paragraphs", ("evilflowers.text.paragraphs",)),
        ("evilflowers.text.tables, evilflowers_text_worker", ("evilflowers.text.tables", "evilflowers_text_worker")),
        ("", ("evilflowers.text.tables", "evilflowers.text.paragraphs", "evilflowers_text_worker")),
    ],
)
def test_text_queue_names_the_consumed_queues(monkeypatch, value, queues):
    monkeypatch.setenv("TEXT_QUEUE", value)
    assert Settings(_env_file=None).consumed_queues == queues


def test_the_search_service_and_query_tasks_are_gone():
    app.loader.import_default_modules()
    for name in ("process_pdf", "delete_document", "stats", "embed_query"):
        assert f"evilflowers_text_worker.{name}" not in app.tasks


def test_extraction_jobs_accept_arguments_a_newer_analyzer_adds():
    app.loader.import_default_modules()
    for name in TASKS:
        parameters = inspect.signature(app.tasks[name].run).parameters.values()
        assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        assert app.tasks[name].acks_late


def test_visibility_timeout_outlasts_every_task_on_the_shared_redis_db(monkeypatch):
    # 12 h, like the catalog and the analyzer: an equation pass on the same db took over 2 h.
    monkeypatch.delenv("BROKER_VISIBILITY_TIMEOUT", raising=False)
    assert Settings(_env_file=None).broker_visibility_timeout == 43200
    monkeypatch.setenv("BROKER_VISIBILITY_TIMEOUT", "600")
    assert Settings(_env_file=None).broker_visibility_timeout == 600


def test_soft_shutdown_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("WORKER_SOFT_SHUTDOWN_TIMEOUT", "30")
    monkeypatch.setenv("WORKER_ENABLE_SOFT_SHUTDOWN_ON_IDLE", "true")
    config = CeleryConfig(Settings(_env_file=None))
    assert config.worker_soft_shutdown_timeout == 30.0 and config.worker_enable_soft_shutdown_on_idle is True


def test_sigterm_stays_warm_under_the_solo_pool(monkeypatch):
    installed = []
    monkeypatch.setattr(worker_module, "install_worker_term_hard_handler", lambda *a, **kw: installed.append(kw))
    requeue_on_sigterm(SimpleNamespace(pool=SoloPool(), controller=object()))
    assert installed == []
    requeue_on_sigterm(SimpleNamespace(pool=object(), controller=object()))
    assert installed == [{"sig": "SIGTERM"}]


def test_extraction_jobs_retry_when_their_worker_process_dies():
    app.loader.import_default_modules()
    for name in TASKS:
        task = app.tasks[name]
        assert task.retry_on_worker_lost and task.max_retries == extraction.LOST_RETRIES


def test_a_job_lost_on_every_try_is_reported_failed(monkeypatch):
    reported = []

    class Job:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def report_failure(self, error):
            reported.append((self.kwargs, error))

    monkeypatch.setitem(extraction.JOBS, "evilflowers_text_worker.paragraphs", Job)
    paragraphs = SimpleNamespace(name="evilflowers_text_worker.paragraphs")
    lost = WorkerLostError("signal 9")
    extraction.report_lost_job(sender=paragraphs, exception=lost, kwargs={"run_id": "run-1", "source": "a.pdf"})
    # Only a lost worker process, only our jobs.
    extraction.report_lost_job(sender=paragraphs, exception=ValueError("no"), kwargs={"run_id": "run-1"})
    extraction.report_lost_job(sender=SimpleNamespace(name="other.task"), exception=lost, kwargs={})
    assert reported == [({"run_id": "run-1", "source": "a.pdf"}, lost)]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for(condition, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def prefork_worker(tmp_path):
    """A real Redis and a `--pool=prefork --concurrency=1` worker of `sigterm_app`, as the image runs it."""
    if shutil.which("redis-server") is None:
        pytest.skip("needs redis-server")
    redis = pytest.importorskip("redis")
    port = _free_port()
    server = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no", "--dir", str(tmp_path)],
        stdout=subprocess.DEVNULL,
    )
    client = redis.Redis(port=port)
    worker = None
    log_path = tmp_path / "worker.log"
    try:
        assert _wait_for(lambda: _ping(client), 5), "redis-server did not start"
        broker = f"redis://127.0.0.1:{port}/0"
        tests = Path(__file__).parent
        # No telemetry: an unreachable collector would hold the shutdown up for its export timeout.
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("OTEL_", "PYROSCOPE_")) and key not in ("HOST_IP", "REMAP_SIGTERM")
        }
        env |= {"SIGTERM_TEST_BROKER": broker, "PYTHONPATH": str(tests)}
        with log_path.open("w") as log:
            worker = subprocess.Popen(
                [sys.executable, "-m", "celery", "-A", "sigterm_app", "worker", "--pool=prefork", "--concurrency=1",
                 "--loglevel=INFO", "--without-mingle", "--without-gossip"],
                env=env, cwd=tests, stdout=log, stderr=subprocess.STDOUT,
            )  # fmt: skip
        # Sent the way the analyzer sends: a Celery app that knows the queue only by name.
        sender = Celery("analyzer", broker=broker, set_as_current=False)
        yield SimpleNamespace(
            redis=client,
            process=worker,
            send=lambda task: sender.send_task(f"evilflowers_text_worker.{task}", queue="evilflowers.text.test"),
            log=log_path.read_text,
        )
    finally:
        if worker is not None and worker.poll() is None:
            worker.kill()
        server.terminate()
        server.wait(timeout=5)


def test_sigterm_requeues_the_running_job(prefork_worker):
    """Issue #13 against a real Redis: SIGTERM puts a running `acks_late` job back on its queue in seconds."""
    client = prefork_worker.redis
    prefork_worker.send("sleepy")
    assert _wait_for(lambda: client.get("sleepy:started") == b"1", 20), prefork_worker.log()

    started = time.monotonic()
    prefork_worker.process.send_signal(signal.SIGTERM)
    assert _wait_for(lambda: client.llen("evilflowers.text.test") == 1, 15), prefork_worker.log()
    assert time.monotonic() - started < 15
    assert client.hlen("unacked") == 0
    prefork_worker.process.wait(timeout=15)
    assert "Initiating Soft Shutdown" in prefork_worker.log()


def test_a_killed_job_is_retried_then_given_up(prefork_worker):
    """A child killed mid-job (an OOM) is sent again, and after `max_retries` acknowledged: no endless loop."""
    client = prefork_worker.redis
    prefork_worker.send("doomed")
    for attempt in (1, 2):
        assert _wait_for(lambda attempt=attempt: client.get("doomed:started") == str(attempt).encode(), 20), (
            prefork_worker.log()
        )
        os.kill(int(client.get("doomed:pid")), signal.SIGKILL)
    # The retry was the last one: nothing left to run, nothing unacknowledged.
    assert _wait_for(lambda: client.hlen("unacked") == 0 and client.llen("evilflowers.text.test") == 0, 15), (
        prefork_worker.log()
    )
    time.sleep(2)
    assert client.get("doomed:started") == b"2"


def _ping(client) -> bool:
    try:
        return client.ping()
    except Exception:
        return False
