"""
retriever.py — Hybrid retrieval: BM25 + FAISS semantic search + RRF fusion.

Retrieval pipeline (local mode):
  1. BM25 (keyword exact match)          → top candidates  ─┐
  2. FAISS IndexFlatIP (cosine search)   → top candidates  ─┤ RRF fusion
  3. RRF score = 1/(k+rank_bm25) + 1/(k+rank_faiss)       ─┘
  4. Return top-k by RRF score

Vector index:
  - IndexFlatIP = exact inner-product search on L2-normalised vectors
  - Equivalent to cosine similarity, no approximation error
  - C++ / SIMD backend (faiss-cpu), ~10-50x faster than Python cosine loop

Why hybrid:
  - Medical terms (e.g. "Flucelvax", "HBsAg", "MMRV") need exact BM25 match
  - Natural-language questions ("is it safe for...") need semantic retrieval
  - RRF avoids score-scale mismatch between the two methods

SEARCH_PROVIDER=local  → BM25 + FAISS + RRF on local JSON file (default)
SEARCH_PROVIDER=azure  → Azure AI Search hybrid (handles fusion server-side)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
_faiss_index = None          # faiss.IndexFlatIP
_faiss_matrix: np.ndarray | None = None   # normalised vectors (n, dim)
_bm25_cache: Any = None      # BM25Okapi instance


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric boundaries.
    Keeps medical abbreviations intact (e.g. 'MMR', 'HBsAg', 'BCG').
    """
    return re.findall(r"\b\w+\b", text.lower())


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_query(query: str) -> np.ndarray:
    """Embed query and return L2-normalised numpy vector (shape: 1, dim)."""
    client = get_openai_client()
    response = client.embeddings.create(
        input=[query],
        model=get_embedding_model(),
    )
    vec = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    # L2-normalise so IndexFlatIP ≡ cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Chunk loader + index builder ──────────────────────────────────────────────

def _load_chunks() -> list[dict]:
    global _chunks_cache, _faiss_index, _faiss_matrix, _bm25_cache

    if _chunks_cache is not None:
        return _chunks_cache

    chunks_path = Path(LOCAL_CHUNKS_FILE)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Local chunks file not found: {chunks_path}\n"
            "Run: python -m ingestion.embed_and_index data/chunks_raw.json --local"
        )

    with open(chunks_path, encoding="utf-8") as f:
        _chunks_cache = json.load(f)

    # ── Build FAISS index ──────────────────────────────────────────────────────
    import faiss

    raw = np.array(
        [c.get("embedding", []) for c in _chunks_cache], dtype="float32"
    )                                           # shape: (n_chunks, dim)

    # L2-normalise every row so inner product = cosine similarity
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid divide-by-zero
    _faiss_matrix = raw / norms

    dim = _faiss_matrix.shape[1]
    _faiss_index = faiss.IndexFlatIP(dim)       # exact inner-product (= cosine)
    _faiss_index.add(_faiss_matrix)

    # ── Build BM25 index ───────────────────────────────────────────────────────
    from rank_bm25 import BM25Okapi
    corpus = [_tokenize(c["content"]) for c in _chunks_cache]
    _bm25_cache = BM25Okapi(corpus)

    print(
        f"[retriever] Loaded {len(_chunks_cache)} chunks | "
        f"FAISS IndexFlatIP dim={dim} | BM25 ready"
    )
    return _chunks_cache


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_fusion(
    bm25_ranking: list[int],
    faiss_ranking: list[int],
    k: int = 60,
) -> dict[int, float]:
    """Reciprocal Rank Fusion. Returns {chunk_idx: rrf_score}, higher = better."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(faiss_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ── Local retriever (BM25 + FAISS + RRF) ─────────────────────────────────────

def _retrieve_local(query: str, top_k: int) -> list[dict[str, Any]]:
    chunks = _load_chunks()
    candidate_pool = top_k * 4   # wider net before fusion

    assert _bm25_cache is not None and _faiss_index is not None

    # ── BM25 ──────────────────────────────────────────────────────────────────
    query_tokens = _tokenize(query)
    bm25_scores = _bm25_cache.get_scores(query_tokens)
    bm25_ranking: list[int] = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:candidate_pool]

    # ── FAISS semantic search ─────────────────────────────────────────────────
    query_vec = _embed_query(query)                          # (1, dim), normalised
    cosine_scores_top, faiss_indices_top = _faiss_index.search(
        query_vec, candidate_pool
    )                                                        # shapes: (1, pool)
    faiss_ranking: list[int] = faiss_indices_top[0].tolist()
    # Build full cosine score array for result annotation
    cosine_score_map: dict[int, float] = {
        int(idx): float(score)
        for idx, score in zip(faiss_indices_top[0], cosine_scores_top[0])
        if idx >= 0
    }

    # ── RRF fusion ────────────────────────────────────────────────────────────
    rrf_scores = _rrf_fusion(bm25_ranking, faiss_ranking)
    top_indices = sorted(
        rrf_scores, key=lambda i: rrf_scores[i], reverse=True
    )[:top_k]

    # ── Build result list ─────────────────────────────────────────────────────
    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        cosine = cosine_score_map.get(idx, 0.0)
        bm25 = float(bm25_scores[idx])
        rrf = rrf_scores[idx]

        # Drop chunks with near-zero signal in both methods
        if cosine < RETRIEVAL_SIMILARITY_THRESHOLD and bm25 < 0.5:
            continue

        meta = chunk.get("metadata", {})
        results.append({
            "content":      chunk.get("content", ""),
            "source_name":  meta.get("source_name", ""),
            "chapter":      meta.get("chapter", ""),
            "section":      meta.get("section", ""),
            "url":          meta.get("url", ""),
            "breadcrumb":   chunk.get("breadcrumb", ""),
            # Transparency scores
            "score":        round(rrf, 6),
            "score_vector": round(cosine, 4),
            "score_bm25":   round(bm25, 4),
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

    query_vec_np = _embed_query(query)
    query_vector = query_vec_np[0].tolist()

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
            "content":     result.get("content", ""),
            "source_name": result.get("source_name", ""),
            "chapter":     result.get("chapter", ""),
            "section":     result.get("section", ""),
            "url":         result.get("url", ""),
            "breadcrumb":  result.get("breadcrumb", ""),
            "score":       round(float(effective_score), 4),
        })
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict[str, Any]]:
    """Retrieve relevant chunks. Backend chosen by SEARCH_PROVIDER env var."""
    if SEARCH_PROVIDER == "azure":
        return _retrieve_azure(query, top_k)
    return _retrieve_local(query, top_k)
