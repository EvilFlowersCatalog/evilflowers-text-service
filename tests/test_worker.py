"""The task contract with the analyzer: names, queues and argument shapes."""

import inspect

from evilflowers_text.settings import Settings
from evilflowers_text.worker import app

TASKS = {
    "evilflowers_text_worker.tables": "evilflowers_text_worker",
    "evilflowers_text_worker.paragraphs": "evilflowers_text_worker",
}


def test_every_task_is_registered_and_routed():
    app.loader.import_default_modules()
    for name, queue in TASKS.items():
        assert name in app.tasks, name
        assert app.amqp.router.route({}, name)["queue"].name == queue


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
