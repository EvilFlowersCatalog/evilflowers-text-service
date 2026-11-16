"""The task contract with the analyzer: names, queues and argument shapes; the kit's delivery semantics."""

import inspect

import pytest

from evilflowers_text.settings import Settings, get_settings
from evilflowers_text.worker import app
from evilflowers_worker.judges import Judges

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
    monkeypatch.delenv("QUEUE", raising=False)
    assert Settings(_env_file=None).consumed_queues == (
        "evilflowers.text.tables",
        "evilflowers.text.paragraphs",
        "evilflowers_text_worker",
    )
    # The app declares what the settings name, each on a direct exchange of the same name.
    declared = {queue.name: queue.exchange.name for queue in app.conf.task_queues}
    assert list(declared) == list(get_settings().consumed_queues)
    assert all(name == exchange for name, exchange in declared.items())


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


def test_tasks_take_any_keyword_and_follow_the_kits_delivery_semantics():
    """Late ack, requeue on a lost worker (capped by MAX_DELIVERIES), never Celery's own retries."""
    app.loader.import_default_modules()
    for name in TASKS:
        task = app.tasks[name]
        parameters = inspect.signature(task.run).parameters.values()
        assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        assert task.acks_late and task.reject_on_worker_lost
        assert not getattr(task, "retry_on_worker_lost", False)


def test_celerys_own_soft_shutdown_is_off():
    """Scale-in is the kit's `shutdown` (#17); Celery's soft shutdown would redo finished jobs."""
    assert app.conf.worker_soft_shutdown_timeout == 0
    assert not app.conf.worker_enable_soft_shutdown_on_idle


def test_scale_in_settings_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("WORKER_SOFT_SHUTDOWN_TIMEOUT", raising=False)
    defaults = Settings(_env_file=None)
    # Text's grace plus the cancel's must stay under the pod's terminationGracePeriodSeconds (900).
    assert defaults.worker_soft_shutdown_timeout == 10 and defaults.worker_cancel_timeout == 30
    monkeypatch.setenv("WORKER_SOFT_SHUTDOWN_TIMEOUT", "25")
    assert Settings(_env_file=None).worker_soft_shutdown_timeout == 25


DECIDER_ENV = (
    "DECIDER_ENABLED",
    "JUDGES_ENABLED",
    "DECIDER_VISION_URL",
    "DECIDER_TIMEOUT",
    "JUDGE_THRESHOLDS",
    "DECIDER_TABLE_BANDS",
)


def test_decider_settings_come_from_the_environment(monkeypatch):
    """The kit's judge settings, under the names the cluster sets (Deploy-Decision-Models), plus text's bands."""
    for name in DECIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    defaults = Settings(_env_file=None)
    assert not defaults.judges_enabled and defaults.decider_vision_url is None and defaults.decider_timeout == 60
    assert defaults.judge_thresholds == {} and defaults.decider_table_bands is None
    monkeypatch.setenv("DECIDER_ENABLED", "true")
    monkeypatch.setenv("DECIDER_VISION_URL", "http://decider-vision.shared.svc.cluster.local")
    monkeypatch.setenv("DECIDER_TIMEOUT", "90")
    monkeypatch.setenv("JUDGE_THRESHOLDS", '{"is_table": {"keep": 0.8, "hide": 0.9}}')
    monkeypatch.setenv("DECIDER_TABLE_BANDS", "0.4, 0.7")
    settings = Settings(_env_file=None)
    assert settings.judges_enabled and settings.decider_timeout == 90
    assert settings.judge_thresholds["is_table"].hide == 0.9 and settings.decider_table_bands == (0.4, 0.7)
    judges = Judges.from_settings(settings)
    assert judges.decider_vision is not None and judges.decider_vision.url == settings.decider_vision_url
    assert judges.decider is None and judges.ollama is None


def test_an_empty_decider_env_is_off_not_a_crash(monkeypatch):
    # A rollback (`decider.enabled: false`) leaves the variables in the pod, empty.
    for name in DECIDER_ENV:
        monkeypatch.setenv(name, "")
    settings = Settings(_env_file=None)
    assert not settings.judges_enabled and settings.decider_vision_url is None and settings.decider_timeout == 60
    assert settings.judge_thresholds == {} and settings.decider_table_bands is None
    assert Judges.from_settings(settings).decider_vision is None


def test_decider_bands_must_be_ordered(monkeypatch):
    monkeypatch.setenv("DECIDER_TABLE_BANDS", "0.7,0.4")
    with pytest.raises(ValueError, match="DECIDER_TABLE_BANDS"):
        Settings(_env_file=None)


def test_visibility_timeout_outlasts_every_task_on_the_shared_redis_db(monkeypatch):
    # 12 h, like the catalog and the analyzer: an equation pass on the same db took over 2 h.
    monkeypatch.delenv("BROKER_VISIBILITY_TIMEOUT", raising=False)
    assert Settings(_env_file=None).broker_visibility_timeout == 43200
    assert app.conf.broker_transport_options["visibility_timeout"] == get_settings().broker_visibility_timeout
