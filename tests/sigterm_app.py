"""A worker for the shutdown tests in `test_worker`: the service's config and two slow tasks.

Each task, once running, records its start and its process id in Redis, so a test
signals a job that is really running, not a message still waiting to be picked up.
"""

import os
import time

import redis

from evilflowers_text.settings import Settings
from evilflowers_text.worker import create_app  # importing it connects `requeue_on_sigterm`

BROKER = os.environ["SIGTERM_TEST_BROKER"]

app = create_app(
    Settings(_env_file=None, redis_url=BROKER, TEXT_QUEUE="evilflowers.text.test", worker_soft_shutdown_timeout=1)
)


def _started(name: str) -> None:
    client = redis.Redis.from_url(BROKER)
    client.incr(f"{name}:started")
    client.set(f"{name}:pid", os.getpid())


@app.task(name="evilflowers_text_worker.sleepy", acks_late=True)
def sleepy() -> None:
    _started("sleepy")
    time.sleep(120)


#: Like the extraction tasks, with no delay before the retry.
@app.task(
    name="evilflowers_text_worker.doomed",
    acks_late=True,
    retry_on_worker_lost=True,
    max_retries=1,
    default_retry_delay=0,
)
def doomed() -> None:
    _started("doomed")
    time.sleep(120)
