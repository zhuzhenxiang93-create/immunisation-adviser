"""
clean_chunks.py — Remove garbled navigation labels from chunk content.

The scraper picked up bilingual UI elements from the website:
    é¡µé¢æ é¢ï¼<title>      (= 页面标题：<title>)
    ç« èï¼<section>         (= 章节：<section>)
    ---

These appear mid-content wherever a page section boundary was encountered.
This script strips them from the existing JSON without re-embedding.

Usage:
    conda run -n immunisation-adviser python -m ingestion.clean_chunks
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CHUNKS_FILE = Path("data/chunks_with_embeddings.json")
BACKUP_FILE = Path("data/chunks_with_embeddings_backup.json")

# Matches the garbled navigation block and everything up to the --- separator
# Pattern: one garbled line, optional second garbled line, then ---
_GARBLED = re.compile(
    r'\n?[é][^\n]*\n(?:[ç][^\n]*\n)?-{3}\n?',
    re.MULTILINE,
)


def clean_content(text: str) -> str:
    cleaned = _GARBLED.sub("\n", text)
    # Collapse multiple blank lines left behind
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def main() -> None:
    print(f"Loading {CHUNKS_FILE} ...")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"  {len(chunks)} chunks loaded")

    # Backup original
    if not BACKUP_FILE.exists():
        print(f"  Backing up to {BACKUP_FILE} ...")
        with open(BACKUP_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

    changed = 0
    for chunk in chunks:
        original = chunk.get("content", "")
        cleaned = clean_content(original)
        if cleaned != original:
            chunk["content"] = cleaned
            changed += 1

        # Also clean garbled section metadata
        meta = chunk.get("metadata", {})
        if meta.get("section") and _GARBLED.search(meta["section"]):
            meta["section"] = ""
        if chunk.get("breadcrumb") and _GARBLED.search(chunk["breadcrumb"]):
            chunk["breadcrumb"] = re.sub(r'\s*>[^>]*[é][^>]*', '', chunk["breadcrumb"]).strip(" >")

    print(f"  {changed} chunks cleaned")

    print(f"  Saving cleaned file ...")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
