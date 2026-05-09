"""
Walk shared storage (catalogs/**/*.pdf) and publish evilflowers_text_worker.process_pdf tasks.

Run from the text-service image root (/app), e.g.:

  python src/scripts/enqueue_pdfs_from_storage.py --dry-run
  python src/scripts/enqueue_pdfs_from_storage.py --limit 10

Uses STORAGE_PATH (default /mnt/data) and REDIS_URL from env — same as the Celery worker.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from uuid import UUID

from celery import Celery

# So `python src/scripts/enqueue_pdfs_from_storage.py` finds `config` without PYTHONPATH
_src_root = Path(__file__).resolve().parents[1]
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from config.Config import Config  # noqa: E402

_CATALOGS = "catalogs"
_TASK = "evilflowers_text_worker.process_pdf"
_QUEUE = "evilflowers_text_worker"


def _celery_app() -> Celery:
    return Celery(
        "text_service",
        broker=Config.REDIS_URL,
        backend=Config.REDIS_URL,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be sent")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument("--catalog-slug", type=str, default=None, metavar="SLUG")
    parser.add_argument(
        "--one-per-entry",
        action="store_true",
        help="At most one task per entry UUID (first path wins)",
    )
    parser.add_argument(
        "--include-encrypted",
        action="store_true",
        help="Include PDFs under .../encrypted/ (Readium LCP)",
    )
    args = parser.parse_args()

    root = Path(os.environ.get("STORAGE_PATH", "/mnt/data"))
    catalogs_dir = root / _CATALOGS
    if not catalogs_dir.is_dir():
        print(f"error: not a directory: {catalogs_dir}", file=sys.stderr)
        print("Set STORAGE_PATH to the volume root that contains catalogs/", file=sys.stderr)
        return 1

    app = _celery_app()
    slug_filter = args.catalog_slug
    seen_entries: set[str] = set()
    enqueued = skipped = failed = 0
    attempt = 0

    for path in sorted(catalogs_dir.rglob("*.pdf")):
        if args.limit is not None and attempt >= args.limit:
            break

        try:
            rel = path.relative_to(root)
        except ValueError:
            skipped += 1
            print(f"skip outside storage root: {path}", file=sys.stderr)
            continue

        rel_posix = rel.as_posix()
        parts = rel_posix.split("/")
        if len(parts) < 4 or parts[0] != _CATALOGS:
            skipped += 1
            print(f"skip unexpected layout: {rel_posix}", file=sys.stderr)
            continue

        if len(parts) >= 5 and parts[3] == "encrypted" and not args.include_encrypted:
            skipped += 1
            print(f"skip encrypted subtree: {rel_posix}", file=sys.stderr)
            continue

        slug = parts[1]
        if slug_filter is not None and slug != slug_filter:
            continue

        try:
            entry_uuid = UUID(parts[2])
        except ValueError:
            skipped += 1
            print(f"skip entry segment not UUID: {rel_posix}", file=sys.stderr)
            continue

        entry_id = str(entry_uuid)
        if args.one_per_entry:
            if entry_id in seen_entries:
                continue
            seen_entries.add(entry_id)

        attempt += 1

        if args.dry_run:
            enqueued += 1
            print(f"would enqueue entry={entry_id} source={rel_posix}")
            continue

        try:
            result = app.send_task(_TASK, args=[rel_posix, entry_id], queue=_QUEUE)
            tid = getattr(result, "id", None)
            if tid:
                enqueued += 1
                print(f"enqueued task_id={tid} entry={entry_id} source={rel_posix}")
            else:
                failed += 1
                print(f"failed to enqueue entry={entry_id} source={rel_posix}", file=sys.stderr)
        except Exception as e:
            failed += 1
            print(f"failed entry={entry_id} source={rel_posix}: {e}", file=sys.stderr)

    print(f"done: enqueued={enqueued} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
