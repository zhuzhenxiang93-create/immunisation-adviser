"""
csv_to_chunks.py — Convert immunisation_rag_chunks CSV to the JSON format
expected by embed_and_index.py.

Overlap strategy:
  The CSV is pre-chunked by section. When a section spans multiple consecutive
  chunks, the last OVERLAP_CHARS characters of the previous chunk are prepended
  to the current chunk so that context is not lost at chunk boundaries.

Usage:
    python -m ingestion.csv_to_chunks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

INPUT_CSV = "data/immunisation_rag_chunks(1).csv"
OUTPUT_JSON = "data/chunks_raw.json"
SOURCE_NAME = "NZ Immunisation Handbook 2024"
OVERLAP_CHARS = 200  # chars from previous chunk to prepend as context


def csv_to_chunks(csv_path: str = INPUT_CSV, output_path: str = OUTPUT_JSON) -> None:
    df = pd.read_csv(csv_path, encoding="latin-1")
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Build chunks without overlap first, preserving original order
    raw_chunks = []
    for _, row in df.iterrows():
        content = str(row["content"]).strip()
        if not content:
            continue

        source = str(row.get("source", SOURCE_NAME)).strip() or SOURCE_NAME
        url = str(row.get("url", "")).strip()
        page_title = str(row.get("page_title", "")).strip()
        section_heading = str(row.get("section_heading", "")).strip()
        chunk_index = int(row["id"]) if "id" in row else len(raw_chunks)

        breadcrumb_parts = [p for p in [source, page_title, section_heading] if p]
        breadcrumb = " > ".join(breadcrumb_parts)

        raw_chunks.append({
            "content": content,
            "metadata": {
                "source_name": source,
                "url": url,
                "chapter": page_title,
                "section": section_heading,
                "chunk_index": chunk_index,
            },
            "breadcrumb": breadcrumb,
        })

    # Add overlap: for consecutive chunks in the same section, prepend the
    # tail of the previous chunk so boundary context is preserved.
    overlap_added = 0
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            chunks.append(chunk)
            continue

        prev = raw_chunks[i - 1]
        same_section = (
            chunk["metadata"]["chapter"] == prev["metadata"]["chapter"]
            and chunk["metadata"]["section"] == prev["metadata"]["section"]
        )

        if same_section:
            tail = prev["content"][-OVERLAP_CHARS:]
            chunk = dict(chunk)  # shallow copy to avoid mutating raw_chunks
            chunk["content"] = tail + "\n\n" + chunk["content"]
            overlap_added += 1

        chunks.append(chunk)

    print(f"Applied overlap to {overlap_added} chunks (same-section boundaries)")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(chunks)} chunks -> {output_path}")


if __name__ == "__main__":
    csv_to_chunks()
