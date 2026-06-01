"""
run_eval.py — Two-layer evaluation of the Immunisation Guidelines Adviser.

Layer 1 — Retrieval quality (automatic, section/source matching):
  Hit Rate @k   : % of questions where correct source appears in top-k chunks
  Recall @k     : same (binary per question)
  MRR           : Mean Reciprocal Rank of first matching source
  Section Hit   : % where correct section keywords appear in top-k chunks

Layer 2 — Generation quality:
  Escalation accuracy : % of not_found questions correctly refused
  Confidence match    : % where predicted confidence == expected confidence
  Citation coverage   : % of non-escalated answers that have ≥1 citation
  Confidence distribution summary
  Human review fields (filled manually after running)

Usage:
  conda activate immunisation-adviser
  python -m evaluation.run_eval
  python -m evaluation.run_eval --sets hand_crafted,transcript
  python -m evaluation.run_eval --output evaluation/results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.query_handler import run_query

QUESTION_SETS = {
    "hand_crafted": Path(__file__).parent / "question_set.json",
    "transcript":   Path(__file__).parent / "transcript_question_set.json",
}

K = 8  # top-k chunks evaluated (matches RETRIEVAL_TOP_K)

# ── Source / section matching ─────────────────────────────────────────────────

_STOPWORDS = {"the", "a", "an", "of", "for", "and", "or", "in", "on", "to",
              "is", "are", "chapter", "section"}

def _keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _source_match(chunk: dict, gt: dict) -> bool:
    """True if chunk source_name contains the key words of gt source."""
    if not gt or not gt.get("source"):
        return False
    gt_words  = set(_keywords(gt["source"]))
    ck_words  = set(_keywords(chunk.get("source_name", "")))
    if not gt_words:
        return False
    overlap = gt_words & ck_words
    return len(overlap) >= max(1, len(gt_words) // 2)


def _section_match(chunk: dict, gt: dict) -> bool:
    """True if chunk metadata contains key words from gt section."""
    if not gt or not gt.get("section"):
        return False
    gt_words = _keywords(gt["section"])
    if not gt_words:
        return False
    chunk_text = " ".join([
        chunk.get("chapter",   ""),
        chunk.get("section",   ""),
        chunk.get("breadcrumb",""),
        chunk.get("content",   "")[:200],
    ]).lower()
    matches = sum(1 for w in gt_words if w in chunk_text)
    return matches >= max(1, len(gt_words) // 2)


# ── Per-question retrieval metrics ────────────────────────────────────────────

def _retrieval_metrics(chunks: list[dict], gt_citation: dict) -> dict:
    """
    For one question compute:
      source_hit_rank  : rank (1-based) of first source match, None if not found
      section_hit_rank : rank (1-based) of first section match, None if not found
      source_hit@k     : bool — source match in top-k
      section_hit@k    : bool — section match in top-k
      rr               : reciprocal rank for source (0 if not found)
    """
    source_rank  = None
    section_rank = None

    for rank, chunk in enumerate(chunks[:K], start=1):
        if source_rank  is None and _source_match(chunk, gt_citation):
            source_rank = rank
        if section_rank is None and _section_match(chunk, gt_citation):
            section_rank = rank

    return {
        "source_hit_rank":  source_rank,
        "section_hit_rank": section_rank,
        f"source_hit@{K}":  source_rank is not None,
        f"section_hit@{K}": section_rank is not None,
        "rr":               1.0 / source_rank if source_rank else 0.0,
    }


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def _aggregate(results: list[dict], k: int = K) -> dict:
    total       = len(results)
    has_gt      = [r for r in results if r.get("ground_truth_citation", {}).get("source")]
    not_found_q = [r for r in results if r.get("expected_confidence") == "not_found"]
    non_nf      = [r for r in results if r.get("expected_confidence") != "not_found"]

    # ── Layer 1 — Retrieval ───────────────────────────────────────────────────
    if has_gt:
        source_hits  = [r["retrieval"][f"source_hit@{k}"]  for r in has_gt]
        section_hits = [r["retrieval"][f"section_hit@{k}"] for r in has_gt]
        rrs          = [r["retrieval"]["rr"] for r in has_gt]
        hit_rate     = sum(source_hits)  / len(has_gt)
        section_rate = sum(section_hits) / len(has_gt)
        mrr          = sum(rrs)          / len(has_gt)
    else:
        hit_rate = section_rate = mrr = 0.0

    # ── Layer 2 — Generation ──────────────────────────────────────────────────
    # Escalation: not_found questions correctly refused
    if not_found_q:
        esc_correct  = sum(1 for r in not_found_q if r["confidence"] == "not_found")
        esc_accuracy = esc_correct / len(not_found_q)
    else:
        esc_correct = esc_accuracy = 0

    # Confidence match rate (across all questions)
    conf_match = sum(
        1 for r in results
        if r.get("confidence") == r.get("expected_confidence")
    )
    conf_match_rate = conf_match / total if total else 0

    # Citation coverage (non-not_found answers with ≥1 citation)
    answered = [r for r in results if r["confidence"] != "not_found"]
    cit_coverage = (
        sum(1 for r in answered if r.get("citations")) / len(answered)
        if answered else 0
    )

    # Confidence distribution
    conf_dist = dict(Counter(r["confidence"] for r in results))

    return {
        "total_questions":   total,
        "questions_with_gt": len(has_gt),
        "layer1_retrieval": {
            f"hit_rate_source@{k}":  round(hit_rate,   3),
            f"hit_rate_section@{k}": round(section_rate,3),
            "MRR":                   round(mrr, 3),
            "n_evaluated":           len(has_gt),
        },
        "layer2_generation": {
            "escalation_accuracy":   round(esc_accuracy, 3),
            "escalation_correct":    esc_correct,
            "escalation_total":      len(not_found_q),
            "confidence_match_rate": round(conf_match_rate, 3),
            "citation_coverage":     round(cit_coverage, 3),
            "confidence_distribution": conf_dist,
        },
        "manual_review_pending": sum(1 for r in results if r["human_correct"] is None),
    }


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(
    sets:        list[str] = ("hand_crafted", "transcript"),
    output_path: str | None = "evaluation/results.json",
    summary_path:str | None = "evaluation/summary.json",
    delay:       float = 1.0,
) -> dict:

    questions: list[dict] = []
    for name in sets:
        path = QUESTION_SETS.get(name)
        if not path or not path.exists():
            print(f"[warn] Question set '{name}' not found at {path}, skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            qs = json.load(f)
        print(f"Loaded {len(qs):3d} questions from {name}")
        questions.extend(qs)

    print(f"\nTotal: {len(questions)} questions | K={K}\n{'='*60}\n")

    results: list[dict] = []

    for i, q in enumerate(questions, 1):
        qid   = q["id"]
        query = q["query"]
        gt    = q.get("ground_truth_citation", {})
        expected_conf = q.get("expected_confidence", "")

        print(f"[{i:02d}/{len(questions)}] {qid} — {query[:70]}...")

        state  = run_query(query)
        output = state.get("output", {})
        chunks = state.get("chunks", [])        # raw retrieved chunk list

        confidence = output.get("confidence", "not_found")
        citations  = output.get("citations", [])
        answer     = output.get("answer", "")

        # Layer 1 — retrieval metrics (skip for not_found expected)
        retrieval = (
            _retrieval_metrics(chunks, gt)
            if gt.get("source") and expected_conf != "not_found"
            else {f"source_hit@{K}": None, f"section_hit@{K}": None,
                  "source_hit_rank": None, "section_hit_rank": None, "rr": None}
        )

        entry = {
            "id":                   qid,
            "source":               q.get("source", "hand_crafted"),
            "category":             q.get("category", ""),
            "query":                query,
            # Generation output
            "confidence":           confidence,
            "answer":               answer,
            "citations":            citations,
            "chunks_retrieved":     len(chunks),
            # Ground truth
            "expected_confidence":  expected_conf,
            "ground_truth_answer":  q.get("ground_truth_answer", ""),
            "ground_truth_citation": gt,
            # Retrieval metrics
            "retrieval":            retrieval,
            # Manual review (fill in after running)
            "human_correct":        None,   # True / False — answer clinically correct?
            "citation_correct":     None,   # True / False — citation supports answer?
            "notes":                "",
        }
        results.append(entry)

        # Console feedback
        src_hit = retrieval.get(f"source_hit@{K}")
        sec_hit = retrieval.get(f"section_hit@{K}")
        print(
            f"  conf={confidence:<10} expected={expected_conf:<10} "
            f"chunks={len(chunks)}  "
            f"src_hit={src_hit}  sec_hit={sec_hit}"
        )
        time.sleep(delay)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = _aggregate(results, K)

    # Save raw results
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved → {out}")

    # Save summary
    if summary_path:
        sp = Path(summary_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved → {sp}")

    _print_summary(summary)
    return summary


# ── Pretty-print summary ──────────────────────────────────────────────────────

def _print_summary(s: dict) -> None:
    l1 = s["layer1_retrieval"]
    l2 = s["layer2_generation"]
    n  = l1["n_evaluated"]

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions : {s['total_questions']}")
    print(f"Questions w/ GT : {s['questions_with_gt']}")
    print()
    print("── Layer 1: Retrieval (source/section matching) ──────────")
    print(f"  Hit Rate  (source@{K}) : {l1[f'hit_rate_source@{K}']:.1%}  ({n} questions)")
    print(f"  Hit Rate (section@{K}) : {l1[f'hit_rate_section@{K}']:.1%}")
    print(f"  MRR                  : {l1['MRR']:.3f}")
    print()
    print("── Layer 2: Generation ───────────────────────────────────")
    print(f"  Escalation accuracy  : {l2['escalation_accuracy']:.1%}"
          f"  ({l2['escalation_correct']}/{l2['escalation_total']} not_found correctly refused)")
    print(f"  Confidence match     : {l2['confidence_match_rate']:.1%}")
    print(f"  Citation coverage    : {l2['citation_coverage']:.1%}")
    print()
    print("  Confidence distribution:")
    for conf, cnt in sorted(l2["confidence_distribution"].items(),
                            key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"    {conf:<12} {cnt:3d}  {bar}")
    print()
    print(f"  Manual review pending: {s['manual_review_pending']} questions")
    print(f"  (Open evaluation/results.json and fill in human_correct / citation_correct)")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the Immunisation Adviser RAG pipeline."
    )
    parser.add_argument(
        "--sets", default="hand_crafted,transcript",
        help="Comma-separated question sets to use (hand_crafted, transcript)"
    )
    parser.add_argument("--output",  default="evaluation/results.json")
    parser.add_argument("--summary", default="evaluation/summary.json")
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between queries (avoid rate limits)"
    )
    args = parser.parse_args()

    run_evaluation(
        sets        = [s.strip() for s in args.sets.split(",")],
        output_path = args.output,
        summary_path= args.summary,
        delay       = args.delay,
    )
