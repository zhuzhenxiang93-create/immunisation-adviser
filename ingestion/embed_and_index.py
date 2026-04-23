"""
embed_and_index.py — Embed chunks and upload to Azure AI Search OR save locally.

Modes:
  --local   Save embeddings to data/chunks_with_embeddings.json  (no Azure needed)
  --azure   Upload to Azure AI Search (requires Azure credentials in .env)

Usage:
  python -m ingestion.embed_and_index chunks.json --local
  python -m ingestion.embed_and_index chunks.json --azure
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.azure_config import (
    get_openai_client,
    get_embedding_model,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME,
)

EMBEDDING_DIMENSIONS = 3072   # text-embedding-3-large
BATCH_SIZE = 16


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embeddings(texts: list[str]) -> list[list[float]]:
    client = get_openai_client()
    response = client.embeddings.create(
        input=texts,
        model=get_embedding_model(),
    )
    return [item.embedding for item in response.data]


# ── Local mode ────────────────────────────────────────────────────────────────

def save_local(chunks: list[dict], output_path: str = "data/chunks_with_embeddings.json") -> None:
    """Embed chunks and save with embeddings to a local JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = len(chunks)
    print(f"Embedding {total} chunks locally in batches of {BATCH_SIZE}...")

    result = []
    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        embeddings = get_embeddings(texts)

        for chunk, embedding in zip(batch, embeddings):
            chunk_with_embedding = dict(chunk)
            chunk_with_embedding["embedding"] = embedding
            result.append(chunk_with_embedding)

        print(f"  Embedded {min(batch_start + BATCH_SIZE, total)}/{total}")
        time.sleep(0.3)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"Saved {len(result)} chunks with embeddings → {out}")


# ── Azure mode ────────────────────────────────────────────────────────────────

def _build_index():
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration, SearchField, SearchFieldDataType,
        SearchIndex, SemanticConfiguration, SemanticField,
        SemanticPrioritizedFields, SemanticSearch, SimpleField,
        SearchableField, VectorSearch, VectorSearchProfile,
    )
    from azure.core.credentials import AzureKeyCredential

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SimpleField(name="source_name", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chapter", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="section", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="url", type=SearchFieldDataType.String),
        SimpleField(name="breadcrumb", type=SearchFieldDataType.String),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-algo")],
    )
    semantic_search = SemanticSearch(configurations=[
        SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")]
            ),
        )
    ])
    client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
    )
    existing = [idx.name for idx in client.list_indexes()]
    if AZURE_SEARCH_INDEX_NAME not in existing:
        client.create_index(SearchIndex(
            name=AZURE_SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        ))
        print(f"Created index: {AZURE_SEARCH_INDEX_NAME}")
    else:
        print(f"Index already exists: {AZURE_SEARCH_INDEX_NAME}")


def upload_chunks(chunks: list[dict]) -> None:
    """Embed and upload chunks to Azure AI Search."""
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential

    _build_index()

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
    )

    total = len(chunks)
    print(f"Uploading {total} chunks to Azure AI Search...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        embeddings = get_embeddings([c["content"] for c in batch])

        documents: list[dict[str, Any]] = []
        for i, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            meta = chunk.get("metadata", {})
            doc: dict[str, Any] = {
                "id": f"{meta.get('source_name','doc').replace(' ','_')}_{batch_start + i}",
                "content": chunk["content"],
                "content_vector": embedding,
                "breadcrumb": chunk.get("breadcrumb", ""),
                "source_name": meta.get("source_name", ""),
                "chapter": meta.get("chapter", ""),
                "section": meta.get("section", ""),
                "url": meta.get("url", ""),
                "chunk_index": meta.get("chunk_index", 0),
            }
            if meta.get("page_number") is not None:
                doc["page_number"] = meta["page_number"]
            documents.append(doc)

        result = search_client.upload_documents(documents=documents)
        succeeded = sum(1 for r in result if r.succeeded)
        print(f"  Batch {batch_start}–{batch_start + len(batch) - 1}: {succeeded}/{len(batch)} uploaded")
        time.sleep(0.5)

    print("Upload complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("chunks_file", help="JSON file of chunks from chunk_documents.py")
    parser.add_argument("--local", action="store_true", help="Save embeddings locally (no Azure)")
    parser.add_argument("--azure", action="store_true", help="Upload to Azure AI Search")
    parser.add_argument("--output", default="data/chunks_with_embeddings.json",
                        help="Output path for --local mode")
    args = parser.parse_args()

    with open(args.chunks_file, encoding="utf-8") as f:
        chunks = json.load(f)

    if args.azure:
        upload_chunks(chunks)
    else:
        save_local(chunks, args.output)
