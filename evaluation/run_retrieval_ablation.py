"""
Content-based retrieval ablation for the Immunisation Adviser.

Compares BM25-only, vector-only, and hybrid RRF using content-level recall:

  - Recall@K : the retrieved top-K chunks collectively contain the key facts
               needed to answer the question (content match against ground truth
               answer text, not section metadata labels).
  - MRR      : reciprocal rank of the first chunk that contains relevant content.

This approach treats the retrieved chunks as a single knowledge pool.
A question is counted as "hit" if ANY chunk in the top-K contains enough
of the key clinical terms from the ground truth answer.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import retriever

K = 10
QUESTION_SET = Path(__file__).parent / "question_set.json"
OUT_JSON = Path(__file__).parent / "ablation_results.json"
OUT_MD   = Path(__file__).parent / "ablation_summary.md"

_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "in", "on", "to", "is",
    "are", "was", "were", "be", "been", "has", "have", "had", "will",
    "should", "may", "can", "not", "with", "from", "this", "that",
    "which", "who", "what", "when", "where", "how", "new", "zealand",
}

# Minimum fraction of GT keywords that must appear in a chunk to count as a hit
CONTENT_MATCH_THRESHOLD = 0.4


def _keywords(text: str) -> list[str]:
    """Extract meaningful clinical keywords from text."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 3]


def _content_match(chunk: dict, gt_answer: str) -> bool:
    """
    True if the chunk content contains enough key terms from the ground truth answer.
    Treats retrieved chunks as a knowledge pool — any chunk that contains
    the relevant clinical facts counts as a hit.
    """
    gt_words = _keywords(gt_answer)
    if not gt_words:
        return False

    content = chunk.get("content", "").lower()
    matches = sum(1 for w in gt_words if w in content)
    required = max(2, int(len(gt_words) * CONTENT_MATCH_THRESHOLD))
    return matches >= required


def _as_result(idx: int, chunks: list[dict]) -> dict:
    chunk = chunks[idx]
    meta = chunk.get("metadata", {})
    return {
        "content":     chunk.get("content", ""),
        "source_name": meta.get("source_name", ""),
        "chapter":     meta.get("chapter", ""),
        "section":     meta.get("section", ""),
        "url":         meta.get("url", ""),
        "breadcrumb":  chunk.get("breadcrumb", ""),
    }


def _rankings(query: str, k: int = K) -> dict[str, list[dict]]:
    chunks = retriever._load_chunks()
    candidate_pool = k * 4

    assert retriever._bm25_cache is not None
    assert retriever._faiss_index is not None

    query_tokens = retriever._tokenize(query)
    bm25_scores  = retriever._bm25_cache.get_scores(query_tokens)
    bm25_ranking = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:candidate_pool]

    query_vec = retriever._embed_query(query)
    _, faiss_indices = retriever._faiss_index.search(query_vec, candidate_pool)
    vector_ranking = [int(i) for i in faiss_indices[0].tolist() if i >= 0]

    rrf_scores     = retriever._rrf_fusion(bm25_ranking, vector_ranking)
    hybrid_ranking = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

    return {
        "bm25_only":   [_as_result(i, chunks) for i in bm25_ranking[:k]],
        "vector_only": [_as_result(i, chunks) for i in vector_ranking[:k]],
        "hybrid_rrf":  [_as_result(i, chunks) for i in hybrid_ranking[:k]],
    }


def _metrics(chunks: list[dict], gt_answer: str) -> dict:
    """
    Recall@K and MRR based on content matching against the ground truth answer.
    The first chunk whose content matches the GT answer determines the rank.
    """
    first_hit_rank = None
    for rank, chunk in enumerate(chunks[:K], start=1):
        if _content_match(chunk, gt_answer):
            first_hit_rank = rank
            break
    return {
        "first_hit_rank": first_hit_rank,
        f"hit@{K}": first_hit_rank is not None,
        "rr": 1.0 / first_hit_rank if first_hit_rank else 0.0,
    }


def main() -> None:
    questions = json.load(open(QUESTION_SET, encoding="utf-8"))

    # Include all answerable questions that have a ground truth answer
    eval_questions = [
        q for q in questions
        if q.get("expected_confidence") != "not_found"
        and q.get("ground_truth_answer", "").strip()
    ]

    print(f"Evaluating {len(eval_questions)} questions (content-based recall)")

    rows = []
    method_names = ["bm25_only", "vector_only", "hybrid_rrf"]
    aggregate = {name: {"hits": 0, "rr_sum": 0.0, "n": 0} for name in method_names}

    for q in eval_questions:
        rankings     = _rankings(q["query"], K)
        gt_answer    = q["ground_truth_answer"]
        method_metrics = {}

        for name, chunks in rankings.items():
            m = _metrics(chunks, gt_answer)
            method_metrics[name] = m
            aggregate[name]["hits"]   += int(m[f"hit@{K}"])
            aggregate[name]["rr_sum"] += m["rr"]
            aggregate[name]["n"]      += 1

        rows.append({
            "id":               q["id"],
            "category":         q.get("category", ""),
            "query":            q["query"],
            "ground_truth_answer": gt_answer,
            "methods":          method_metrics,
        })

    # Build summary
    summary = {}
    for name, values in aggregate.items():
        n = values["n"]
        summary[name] = {
            "hit_count":     values["hits"],
            "n":             n,
            f"recall@{K}":   round(values["hits"] / n, 3) if n else 0.0,
            "MRR":           round(values["rr_sum"] / n, 3) if n else 0.0,
        }

    payload = {
        "metric":       f"content_recall@{K}",
        "match_threshold": CONTENT_MATCH_THRESHOLD,
        "n_evaluated":  len(eval_questions),
        "summary":      summary,
        "results":      rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Content-Based Retrieval Recall",
        "",
        f"Evaluated {len(eval_questions)} questions.",
        f"Metric: Recall@{K} — at least {int(CONTENT_MATCH_THRESHOLD*100)}% of ground-truth answer",
        f"keywords must appear in any one retrieved chunk.",
        "",
        f"| Method | Recall@{K} | Rate | MRR |",
        "|---|---:|---:|---:|",
    ]
    for name in method_names:
        s = summary[name]
        label = name.replace("_", " ")
        lines.append(
            f"| {label} | {s['hit_count']}/{s['n']} | "
            f"{s[f'recall@{K}']:.1%} | {s['MRR']:.3f} |"
        )
    lines.extend([
        "",
        f"Note: A question is counted as hit if any of the top-{K} retrieved chunks",
        "contains the key clinical facts from the ground truth answer.",
        "This is a content-level check, not a section metadata match.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
