"""
chunk_documents.py — Split parsed text into chunks with full metadata.

Each chunk preserves:
  - source_name   : e.g. "NZ Immunisation Handbook"
  - chapter       : chapter title
  - section       : section title
  - url           : original URL or file path
  - page_number   : page number where applicable
  - chunk_index   : position within this document

Usage:
    from ingestion.chunk_documents import chunk_document
    chunks = chunk_document(text, metadata)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

# SKILL.md: 512–1024 tokens, ~10% overlap
# Using characters as proxy: ~4 chars/token → 800 tokens ≈ 3200 chars
CHUNK_SIZE = 1000        # tokens (approximated as characters / 4)
CHUNK_OVERLAP = 100      # ~10% overlap


@dataclass
class ChunkMetadata:
    source_name: str
    url: str
    chapter: str = ""
    section: str = ""
    page_number: Optional[int] = None
    chunk_index: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def chunk_document(
    text: str,
    source_name: str,
    url: str,
    chapter: str = "",
    section: str = "",
    page_number: Optional[int] = None,
) -> list[dict]:
    """
    Split a single document section into overlapping chunks.
    Returns a list of dicts ready to upload to Azure AI Search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,      # convert token estimate to chars
        chunk_overlap=CHUNK_OVERLAP * 4,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    result = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        meta = ChunkMetadata(
            source_name=source_name,
            url=url,
            chapter=chapter,
            section=section,
            page_number=page_number,
            chunk_index=i,
        )

        result.append({
            "content": chunk_text,
            "metadata": meta.to_dict(),
            # Breadcrumb for display in citations
            "breadcrumb": _build_breadcrumb(source_name, chapter, section),
        })

    return result


def _build_breadcrumb(source_name: str, chapter: str, section: str) -> str:
    """e.g. 'NZ Immunisation Handbook > Chapter 4 — Influenza > Storage'"""
    parts = [p for p in [source_name, chapter, section] if p]
    return " > ".join(parts)


# ---------------------------------------------------------------------------
# Helper: extract chapter/section headings from plain text
# ---------------------------------------------------------------------------

def extract_sections(text: str) -> list[dict]:
    """
    Naively split a document on Markdown-style or numbered headings.
    Returns list of {heading, body} dicts.
    Falls back to returning the whole text as one section.
    """
    heading_pattern = re.compile(
        r"^(#{1,3}\s+.+|[0-9]+\.[0-9]*\s+[A-Z].+)$", re.MULTILINE
    )
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return [{"heading": "", "body": text}]

    sections = []
    for idx, match in enumerate(matches):
        heading = match.group(0).strip("#").strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append({"heading": heading, "body": body})

    return sections
