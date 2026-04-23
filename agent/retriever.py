"""
retriever.py — Hybrid retrieval: BM25 + Vector + RRF fusion.

Retrieval pipeline (local mode):
  1. BM25 (keyword exact match)  → top-2k candidates  ─┐
  2. Vector (cosine similarity)  → top-2k candidates  ─┤ RRF fusion
  3. RRF score = 1/(k+rank_bm25) + 1/(k+rank_vector)  ─┘
  4. Return top-k by RRF score

Why RRF:
  - Medical terms (e.g. "Flucelvax", "HBsAg", "MMRV") benefit from exact BM25 match
  - Semantic queries ("is it safe for...") benefit from vector search
  - RRF avoids score-scale mismatch between the two methods

SEARCH_PROVIDER=local  → BM25 + Vector + RRF on local JSON file (default)
SEARCH_PROVIDER=azure  → Azure AI Search hybrid (handles fusion server-side)
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.azure_config import (
    SEARCH_PROVIDER,
    LOCAL_CHUNKS_FILE,
    RETRIEVAL_TOP_K,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    get_openai_client,
    get_embedding_model,
)

# ── In-memory cache (rebuilt once per process) ───────────────────────────────
_chunks_cache: list[dict] | None = None
_bm25_cache: Any = None          # BM25Okapi instance
_corpus_cache: list[list[str]] | None = None


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Lowercase + split on non-alphanumeric boundaries.
    Keeps medical abbreviations intact (e.g. 'MMR', 'HBsAg', 'BCG').
    """
    return re.findall(r"\b\w+\b", text.lower())


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_query(query: str) -> list[float]:
    client = get_openai_client()
    response = client.embeddings.create(
        input=[query],
        model=get_embedding_model(),
    )
    return response.data[0].embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Chunk loader ──────────────────────────────────────────────────────────────

def _load_chunks() -> list[dict]:
    global _chunks_cache, _bm25_cache, _corpus_cache
    if _chunks_cache is not None:
        return _chunks_cache

    chunks_path = Path(LOCAL_CHUNKS_FILE)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Local chunks file not found: {chunks_path}\n"
            "Run: python -m ingestion.embed_and_index data/sample_chunks.json --local"
        )

    with open(chunks_path, encoding="utf-8") as f:
        _chunks_cache = json.load(f)

    # Build BM25 index once
    from rank_bm25 import BM25Okapi
    _corpus_cache = [_tokenize(c["content"]) for c in _chunks_cache]
    _bm25_cache = BM25Okapi(_corpus_cache)
    print(f"[retriever] Loaded {len(_chunks_cache)} chunks, BM25 index built.")

    return _chunks_cache


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_fusion(
    bm25_ranking: list[int],
    vector_ranking: list[int],
    k: int = 60,
) -> dict[int, float]:
    """
    Reciprocal Rank Fusion.
    Returns {chunk_index: rrf_score}, higher is better.
    """
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(vector_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ── Local retriever (BM25 + Vector + RRF) ────────────────────────────────────

def _retrieve_local(query: str, top_k: int) -> list[dict[str, Any]]:
    chunks = _load_chunks()
    candidate_pool = top_k * 4   # wider net before fusion

    assert _bm25_cache is not None, "BM25 index not built"

    # ── BM25 ──────────────────────────────────────────────────────────────────
    query_tokens = _tokenize(query)
    bm25_scores = _bm25_cache.get_scores(query_tokens)
    bm25_ranking = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:candidate_pool]

    # ── Vector ────────────────────────────────────────────────────────────────
    query_vec = _embed_query(query)
    cosine_scores = [
        _cosine_similarity(query_vec, c.get("embedding", []))
        for c in chunks
    ]
    vector_ranking = sorted(
        range(len(cosine_scores)),
        key=lambda i: cosine_scores[i],
        reverse=True,
    )[:candidate_pool]

    # ── RRF fusion ────────────────────────────────────────────────────────────
    rrf_scores = _rrf_fusion(bm25_ranking, vector_ranking)
    top_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]

    # ── Build result list ─────────────────────────────────────────────────────
    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        cosine = cosine_scores[idx]
        bm25 = float(bm25_scores[idx])
        rrf = rrf_scores[idx]

        # Drop chunks with near-zero signal in both methods
        if cosine < RETRIEVAL_SIMILARITY_THRESHOLD and bm25 < 0.5:
            continue

        meta = chunk.get("metadata", {})
        results.append({
            "content": chunk.get("content", ""),
            "source_name": meta.get("source_name", ""),
            "chapter": meta.get("chapter", ""),
            "section": meta.get("section", ""),
            "url": meta.get("url", ""),
            "breadcrumb": chunk.get("breadcrumb", ""),
            # Scores for transparency
            "score": round(rrf, 6),
            "score_vector": round(cosine, 4),
            "score_bm25": round(bm25, 4),
        })

    return results


# ── Azure AI Search retriever ─────────────────────────────────────────────────

def _retrieve_azure(query: str, top_k: int) -> list[dict[str, Any]]:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery
    from config.azure_config import (
        AZURE_SEARCH_ENDPOINT,
        AZURE_SEARCH_API_KEY,
        AZURE_SEARCH_INDEX_NAME,
    )

    query_vector = _embed_query(query)
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
    )
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )
    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["content", "source_name", "chapter", "section", "url", "breadcrumb"],
        top=top_k,
        query_type="semantic",
        semantic_configuration_name="semantic-config",
    )
    chunks = []
    for result in results:
        score = result.get("@search.score", 0.0)
        reranker_score = result.get("@search.reranker_score")
        effective_score = reranker_score if reranker_score is not None else score
        if reranker_score is None and effective_score < RETRIEVAL_SIMILARITY_THRESHOLD:
            continue
        chunks.append({
            "content": result.get("content", ""),
            "source_name": result.get("source_name", ""),
            "chapter": result.get("chapter", ""),
            "section": result.get("section", ""),
            "url": result.get("url", ""),
            "breadcrumb": result.get("breadcrumb", ""),
            "score": round(float(effective_score), 4),
        })
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict[str, Any]]:
    """Retrieve relevant chunks. Backend chosen by SEARCH_PROVIDER env var."""
    if SEARCH_PROVIDER == "azure":
        return _retrieve_azure(query, top_k)
    return _retrieve_local(query, top_k)
