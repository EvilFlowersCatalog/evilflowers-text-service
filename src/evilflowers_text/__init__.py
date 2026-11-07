"""EvilFlowers text service: the Celery worker that turns a library's PDFs into text, tables and paragraphs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("evilflowers-text-service")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"
