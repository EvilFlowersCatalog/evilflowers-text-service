"""Deprecated entry point: ``celery --workdir src -A main worker`` (deployments that override the command).

Use ``celery -A evilflowers_text.worker worker``; remove this file once no manifest names ``main``.
"""

from evilflowers_text.worker import app  # noqa: F401
