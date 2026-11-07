"""``evilflowers-text``: operator commands next to the worker.

  evilflowers-text enqueue --dry-run              # what would be (re)indexed
  evilflowers-text enqueue --limit 10             # publish process_pdf for catalogs/**/*.pdf
  evilflowers-text download-models                # prefetch weights into HF_HOME (/models)

Uses the worker's settings (``STORAGE_PATH``, ``REDIS_URL`` …).
"""

import argparse
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from evilflowers_text.settings import Settings, get_settings

CATALOGS = "catalogs"
PROCESS_PDF = "evilflowers_text_worker.process_pdf"


@dataclass(frozen=True)
class StoredPdf:
    source: str
    entry_id: str
    catalog: str


def stored_pdfs(root: Path, *, include_encrypted: bool = False) -> Iterator[StoredPdf | str]:
    """Every ``catalogs/<slug>/<entry uuid>/…/*.pdf`` under `root`; a str explains a skipped path."""
    for path in sorted((root / CATALOGS).rglob("*.pdf")):
        source = path.relative_to(root).as_posix()
        parts = source.split("/")
        if len(parts) < 4:
            yield f"skip unexpected layout: {source}"
            continue
        if len(parts) >= 5 and parts[3] == "encrypted" and not include_encrypted:
            yield f"skip encrypted subtree: {source}"
            continue
        try:
            entry_id = str(UUID(parts[2]))
        except ValueError:
            yield f"skip entry segment not UUID: {source}"
            continue
        yield StoredPdf(source, entry_id, parts[1])


def selected(args: argparse.Namespace, root: Path) -> Iterator[StoredPdf | str]:
    """`stored_pdfs` narrowed to `--catalog-slug` and, with `--one-per-entry`, the first file of each entry."""
    seen: set[str] = set()
    for item in stored_pdfs(root, include_encrypted=args.include_encrypted):
        if isinstance(item, StoredPdf):
            if args.catalog_slug is not None and item.catalog != args.catalog_slug:
                continue
            if args.one_per_entry and item.entry_id in seen:
                continue
            seen.add(item.entry_id)
        yield item


def enqueue(args: argparse.Namespace, settings: Settings) -> int:
    root = settings.storage_path
    if not (root / CATALOGS).is_dir():
        print(f"error: not a directory: {root / CATALOGS}", file=sys.stderr)
        print("Set STORAGE_PATH to the volume root that contains catalogs/", file=sys.stderr)
        return 1

    app = None
    if not args.dry_run:
        from evilflowers_text.worker import app

    enqueued = skipped = failed = 0
    for item in selected(args, root):
        if args.limit is not None and enqueued + failed >= args.limit:
            break
        if isinstance(item, str):
            skipped += 1
            print(item, file=sys.stderr)
        elif app is None:
            enqueued += 1
            print(f"would enqueue entry={item.entry_id} source={item.source}")
        else:
            try:
                result = app.send_task(PROCESS_PDF, args=[item.source, item.entry_id], queue=settings.queue)
                enqueued += 1
                print(f"enqueued task_id={result.id} entry={item.entry_id} source={item.source}")
            except Exception as error:
                failed += 1
                print(f"failed entry={item.entry_id} source={item.source}: {error}", file=sys.stderr)

    print(f"done: enqueued={enqueued} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


def download_models(_args: argparse.Namespace, settings: Settings) -> int:
    """Fetch the embedding model and the ToC layout model into the Hugging Face cache."""
    from huggingface_hub import hf_hub_download, snapshot_download

    print(f"embedding model: {snapshot_download(settings.embedding_model)}")
    print(f"layout model: {hf_hub_download(repo_id=settings.yolo_repo_id, filename=settings.yolo_filename)}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="evilflowers-text", description=__doc__.split("\n\n")[0])
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("enqueue", help="publish process_pdf for every stored PDF")
    command.add_argument("--dry-run", action="store_true", help="only print what would be sent")
    command.add_argument("--limit", type=int, default=None, metavar="N")
    command.add_argument("--catalog-slug", default=None, metavar="SLUG")
    command.add_argument("--one-per-entry", action="store_true", help="at most one task per entry (first path wins)")
    command.add_argument("--include-encrypted", action="store_true", help="include .../encrypted/ (Readium LCP)")
    command.set_defaults(handler=enqueue)

    command = commands.add_parser("download-models", help="prefetch model weights into HF_HOME")
    command.set_defaults(handler=download_models)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.handler(args, get_settings())


if __name__ == "__main__":
    raise SystemExit(main())
