"""
Section-level retrieval ablation for the Immunisation Adviser.

Compares BM25-only, vector-only, hybrid RRF, and hybrid RRF + section reranker
using the same section-level metrics as evaluation/run_eval.py:

  - Section Hit Rate @8
  - Mean Reciprocal Rank (MRR)

This intentionally avoids source-level hit rate because source-level matching is
too coarse for citation quality: finding the right handbook or website is less
useful than surfacing the specific guidance section that supports the answer.
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
OUT_MD = Path(__file__).parent / "ablation_summary.md"

_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "in", "on", "to",
    "is", "are", "chapter", "section",
}


def _keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _section_match(chunk: dict, gt: dict) -> bool:
    if not gt or not gt.get("section"):
        return False
    gt_words = _keywords(gt["section"])
    if not gt_words:
        return False

    meta = " ".join([
        chunk.get("chapter", ""),
        chunk.get("section", ""),
        chunk.get("breadcrumb", ""),
    ]).lower()
    matches = sum(1 for w in gt_words if w in meta)
    return matches >= max(1, len(gt_words) // 2)


def _as_result(idx: int, chunks: list[dict]) -> dict:
    chunk = chunks[idx]
    meta = chunk.get("metadata", {})
    return {
        "content": chunk.get("content", ""),
        "source_name": meta.get("source_name", ""),
        "chapter": meta.get("chapter", ""),
        "section": meta.get("section", ""),
        "url": meta.get("url", ""),
        "breadcrumb": chunk.get("breadcrumb", ""),
    }


def _rankings(query: str, k: int = K) -> dict[str, list[dict]]:
    chunks = retriever._load_chunks()
    candidate_pool = k * 4

    assert retriever._bm25_cache is not None
    assert retriever._faiss_index is not None

    query_tokens = retriever._tokenize(query)
    bm25_scores = retriever._bm25_cache.get_scores(query_tokens)
    bm25_ranking = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:candidate_pool]

    query_vec = retriever._embed_query(query)
    _, faiss_indices = retriever._faiss_index.search(query_vec, candidate_pool)
    vector_ranking = [int(i) for i in faiss_indices[0].tolist() if i >= 0]

    rrf_scores = retriever._rrf_fusion(bm25_ranking, vector_ranking)
    hybrid_ranking = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

    return {
        "bm25_only":  [_as_result(i, chunks) for i in bm25_ranking[:k]],
        "vector_only": [_as_result(i, chunks) for i in vector_ranking[:k]],
        "hybrid_rrf": [_as_result(i, chunks) for i in hybrid_ranking[:k]],
    }


def _metrics(chunks: list[dict], gt: dict) -> dict:
    section_rank = None
    for rank, chunk in enumerate(chunks[:K], start=1):
        if _section_match(chunk, gt):
            section_rank = rank
            break
    return {
        "section_rank": section_rank,
        f"hit@{K}": section_rank is not None,
        "rr": 1.0 / section_rank if section_rank else 0.0,
    }


def main() -> None:
    questions = json.load(open(QUESTION_SET, encoding="utf-8"))
    eval_questions = [
        q for q in questions
        if q.get("expected_confidence") != "not_found"
        and q.get("ground_truth_citation", {}).get("section")
        and q.get("ground_truth_citation", {}).get("section") != "N/A"
    ]

    rows = []
    method_names = [
        "bm25_only",
        "vector_only",
        "hybrid_rrf",
    ]
    aggregate = {
        name: {"hits": 0, "rr_sum": 0.0, "n": 0}
        for name in method_names
    }

    for q in eval_questions:
        rankings = _rankings(q["query"], K)
        method_metrics = {}
        for name, chunks in rankings.items():
            m = _metrics(chunks, q["ground_truth_citation"])
            method_metrics[name] = m
            aggregate[name]["hits"] += int(m[f"hit@{K}"])
            aggregate[name]["rr_sum"] += m["rr"]
            aggregate[name]["n"] += 1

        rows.append({
            "id": q["id"],
            "category": q.get("category", ""),
            "query": q["query"],
            "ground_truth_citation": q["ground_truth_citation"],
            "methods": method_metrics,
        })

    summary = {}
    for name, values in aggregate.items():
        n = values["n"]
        summary[name] = {
            "section_hit_count": values["hits"],
            "n": n,
            f"section_hit_rate@{K}": round(values["hits"] / n, 3) if n else 0.0,
            "MRR": round(values["rr_sum"] / n, 3) if n else 0.0,
        }

    payload = {
        "metric": f"section_hit@{K}",
        "n_evaluated": len(eval_questions),
        "summary": summary,
        "results": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Section-Level Retrieval Ablation",
        "",
        f"Evaluated {len(eval_questions)} hand-crafted questions with ground-truth sections.",
        f"Metric: Section Hit@{K} and MRR.",
        "",
        "| Method | Section Hit@8 | Rate | MRR |",
        "|---|---:|---:|---:|",
    ]
    for name in method_names:
        s = summary[name]
        label = name.replace("_", " ")
        lines.append(
            f"| {label} | {s['section_hit_count']}/{s['n']} | "
            f"{s[f'section_hit_rate@{K}']:.1%} | {s['MRR']:.3f} |"
        )
    lines.extend([
        "",
        "Note: This ablation is retrieval-only. It does not judge final answer "
        "correctness or whether generated citations fully support the answer.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
