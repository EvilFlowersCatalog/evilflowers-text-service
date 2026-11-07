"""The task contract with the catalog and the analyzer: names, queues and argument shapes."""

import inspect

from evilflowers_text.worker import app

TASKS = {
    "evilflowers_text_worker.process_pdf": "evilflowers_text_worker",
    "evilflowers_text_worker.delete_document": "evilflowers_text_worker",
    "evilflowers_text_worker.stats": "evilflowers_text_worker",
    "evilflowers_text_worker.tables": "evilflowers_text_worker",
    "evilflowers_text_worker.paragraphs": "evilflowers_text_worker",
    "evilflowers_text_worker.embed_query": "evilflowers_text_embed",
}


def test_every_task_is_registered_and_routed():
    app.loader.import_default_modules()
    for name, queue in TASKS.items():
        assert name in app.tasks, name
        assert app.amqp.router.route({}, name)["queue"].name == queue


def test_extraction_jobs_accept_arguments_a_newer_analyzer_adds():
    app.loader.import_default_modules()
    for name in ("evilflowers_text_worker.tables", "evilflowers_text_worker.paragraphs"):
        parameters = inspect.signature(app.tasks[name].run).parameters.values()
        assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        assert app.tasks[name].acks_late


def test_positional_callers_still_bind():
    app.loader.import_default_modules()
    inspect.signature(app.tasks["evilflowers_text_worker.process_pdf"].run).bind("a/b.pdf", "entry")
    inspect.signature(app.tasks["evilflowers_text_worker.embed_query"].run).bind("query", None)
