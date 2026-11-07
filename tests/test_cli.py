from evilflowers_text.cli import StoredPdf, main, stored_pdfs

ENTRY = "0b52959f-1c2a-4e1f-b44f-4e2c9bd1403e"


def touch(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7")


def test_stored_pdfs_follow_the_catalog_layout(tmp_path):
    touch(tmp_path, f"catalogs/lib/{ENTRY}/book.pdf")
    touch(tmp_path, f"catalogs/lib/{ENTRY}/encrypted/book.pdf")
    touch(tmp_path, "catalogs/lib/not-a-uuid/book.pdf")
    touch(tmp_path, "catalogs/loose.pdf")
    found = list(stored_pdfs(tmp_path))
    assert StoredPdf(f"catalogs/lib/{ENTRY}/book.pdf", ENTRY, "lib") in found
    assert sum(isinstance(item, str) for item in found) == 3
    assert len(list(stored_pdfs(tmp_path, include_encrypted=True))) == 4


def test_enqueue_dry_run(tmp_path, monkeypatch, capsys):
    touch(tmp_path, f"catalogs/lib/{ENTRY}/book.pdf")
    monkeypatch.setenv("STORAGE_PATH", str(tmp_path))
    from evilflowers_text.settings import get_settings

    get_settings.cache_clear()
    try:
        assert main(["enqueue", "--dry-run"]) == 0
    finally:
        get_settings.cache_clear()
    assert f"would enqueue entry={ENTRY}" in capsys.readouterr().out
