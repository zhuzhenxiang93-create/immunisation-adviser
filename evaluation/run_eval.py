"""
run_eval.py — Two-layer evaluation of the Immunisation Guidelines Adviser.

Metrics computed
────────────────
Layer 1 — Retrieval / Citation Quality (automatic):
  Section Hit Rate @k  : % of questions where the correct section appears in
                         top-k retrieved chunks (metadata match only, not content).
                         With 1 GT label per question this equals Recall@k.
                         Directly measures source citation quality.
  MRR                  : Mean Reciprocal Rank — how early the correct section
                         appears in the ranked list.

  NOTE: Source-level hit rate is intentionally omitted — it is trivially high
  because ~2,275 NZ Handbook chunks exist in the KB.
  NOTE: True Recall requires labelling ALL relevant sections per question.
  With 1 GT label, Hit Rate is an honest lower-bound proxy for Recall.

Layer 2 — Generation / Accuracy (automatic + manual):
  Escalation accuracy  : % of not_found questions correctly refused.
  Confidence match     : % where predicted confidence == expected confidence.
  Citation coverage    : % of answered questions with ≥1 citation.
  Confidence distribution: breakdown of high/medium/low/not_found.
  Human correctness    : (manual) answer clinically correct? (fill after run)
  Human citation acc.  : (manual) citation supports answer? (fill after run)

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
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.query_handler import run_query

QUESTION_SETS = {
    "hand_crafted": Path(__file__).parent / "question_set.json",
    "transcript":   Path(__file__).parent / "transcript_question_set.json",
}

K = 10  # matches RETRIEVAL_TOP_K

# ── Section matching ──────────────────────────────────────────────────────────

_STOPWORDS = {"the", "a", "an", "of", "for", "and", "or", "in", "on", "to",
              "is", "are", "chapter", "section"}


def _keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _section_match(chunk: dict, gt: dict) -> bool:
    """
    True if chunk STRUCTURAL METADATA (chapter, section, breadcrumb) contains
    ≥ half the key words from the ground-truth section label.

    Excludes chunk content deliberately — the retriever must surface the
    correct section heading, not just any chunk that mentions the topic.
    This makes Section Hit Rate strictly harder than a topic-level check.
    """
    if not gt or not gt.get("section"):
        return False
    gt_words = _keywords(gt["section"])
    if not gt_words:
        return False
    meta = " ".join([
        chunk.get("chapter",    ""),
        chunk.get("section",    ""),
        chunk.get("breadcrumb", ""),
    ]).lower()
    matches = sum(1 for w in gt_words if w in meta)
    return matches >= max(1, len(gt_words) // 2)


# ── Per-question retrieval metrics ────────────────────────────────────────────

def _retrieval_metrics(chunks: list[dict], gt_citation: dict) -> dict:
    """
    Section Hit Rate @k and MRR for one question.

    With exactly 1 ground-truth section per question:
      Section Hit Rate @k  =  1 if correct section in top-k, else 0
      MRR                  =  1/rank of first section match, else 0
    """
    section_rank = None
    for rank, chunk in enumerate(chunks[:K], start=1):
        if _section_match(chunk, gt_citation):
            section_rank = rank
            break

    return {
        "section_rank":    section_rank,
        f"hit@{K}":        section_rank is not None,
        "rr":              1.0 / section_rank if section_rank else 0.0,
    }


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def _aggregate(results: list[dict], k: int = K) -> dict:
    total       = len(results)
    has_gt      = [r for r in results
                   if r.get("ground_truth_citation", {}).get("section")]
    not_found_q = [r for r in results
                   if r.get("expected_confidence") == "not_found"]
    answered    = [r for r in results if r["confidence"] != "not_found"]

    # ── Layer 1: Retrieval / Citation Quality ─────────────────────────────────
    if has_gt:
        hits = [r["retrieval"][f"hit@{k}"] for r in has_gt]
        rrs  = [r["retrieval"]["rr"]        for r in has_gt]
        section_hit_rate = sum(hits) / len(has_gt)
        mrr              = sum(rrs)  / len(has_gt)
    else:
        section_hit_rate = mrr = 0.0

    # ── Layer 2: Generation / Accuracy ───────────────────────────────────────
    esc_correct  = sum(1 for r in not_found_q if r["confidence"] == "not_found")
    esc_accuracy = esc_correct / len(not_found_q) if not_found_q else 0.0

    conf_match = sum(
        1 for r in results
        if r.get("confidence") == r.get("expected_confidence")
    )
    conf_match_rate = conf_match / total if total else 0.0

    cit_coverage = (
        sum(1 for r in answered if r.get("citations")) / len(answered)
        if answered else 0.0
    )

    # Manual review (filled in after running)
    reviewed       = [r for r in results if r["human_correct"] is not None]
    human_correct  = sum(1 for r in reviewed if r["human_correct"]) / len(reviewed) if reviewed else None
    reviewed_cit   = [r for r in results if r["citation_correct"] is not None]
    human_cit_acc  = sum(1 for r in reviewed_cit if r["citation_correct"]) / len(reviewed_cit) if reviewed_cit else None

    return {
        "total_questions":   total,
        "questions_with_gt": len(has_gt),

        "layer1_citation_quality": {
            f"section_hit_rate@{k}": round(section_hit_rate, 3),
            "MRR":                   round(mrr, 3),
            "n_evaluated":           len(has_gt),
            "note": (
                "Section Hit Rate = lower-bound proxy for Recall@k. "
                "1 GT label per question; multiple relevant sections possible."
            ),
        },

        "layer2_accuracy": {
            "escalation_accuracy":   round(esc_accuracy, 3),
            "escalation_correct":    esc_correct,
            "escalation_total":      len(not_found_q),
            "confidence_match_rate": round(conf_match_rate, 3),
            "citation_coverage":     round(cit_coverage, 3),
            "confidence_distribution": dict(Counter(r["confidence"] for r in results)),
            "human_answer_correctness": round(human_correct, 3) if human_correct is not None else "pending manual review",
            "human_citation_accuracy":  round(human_cit_acc, 3)  if human_cit_acc  is not None else "pending manual review",
            "manual_reviewed":          len(reviewed),
        },
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_evaluation(
    sets:         list[str] = ("hand_crafted", "transcript"),
    output_path:  str | None = "evaluation/results.json",
    summary_path: str | None = "evaluation/summary.json",
    delay:        float = 1.0,
) -> dict:

    questions: list[dict] = []
    for name in sets:
        path = QUESTION_SETS.get(name)
        if not path or not path.exists():
            print(f"[warn] '{name}' not found at {path}, skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            qs = json.load(f)
        print(f"Loaded {len(qs):3d} questions  ← {name}")
        questions.extend(qs)

    print(f"\nTotal: {len(questions)} questions | K={K}\n{'='*60}\n")

    results: list[dict] = []

    for i, q in enumerate(questions, 1):
        qid   = q["id"]
        query = q["query"]
        gt    = q.get("ground_truth_citation", {})
        exp   = q.get("expected_confidence", "")

        print(f"[{i:02d}/{len(questions)}] {qid}  {query[:65]}...")

        state      = run_query(query)
        output     = state.get("output", {})
        chunks     = state.get("chunks", [])
        confidence = output.get("confidence", "not_found")
        citations  = output.get("citations", [])
        answer     = output.get("answer", "")

        retrieval = (
            _retrieval_metrics(chunks, gt)
            if gt.get("section") and exp != "not_found"
            else {"section_rank": None, f"hit@{K}": None, "rr": None}
        )

        entry = {
            "id":                    qid,
            "source":                q.get("source", "hand_crafted"),
            "category":              q.get("category", ""),
            "query":                 query,
            "confidence":            confidence,
            "answer":                answer,
            "citations":             citations,
            "chunks_retrieved":      len(chunks),
            "expected_confidence":   exp,
            "ground_truth_answer":   q.get("ground_truth_answer", ""),
            "ground_truth_citation": gt,
            "retrieval":             retrieval,
            # ── Fill in manually after running ──
            "human_correct":         None,   # True/False: answer clinically correct?
            "citation_correct":      None,   # True/False: citation supports answer?
            "notes":                 "",
        }
        results.append(entry)

        hit = retrieval.get(f"hit@{K}")
        print(f"  conf={confidence:<10}  expected={exp:<10}  "
              f"chunks={len(chunks)}  section_hit={hit}")
        time.sleep(delay)

    summary = _aggregate(results, K)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults  → {out}")

    if summary_path:
        sp = Path(summary_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary  → {sp}")

    _print_summary(summary)
    return summary


# ── Pretty print ──────────────────────────────────────────────────────────────

def _print_summary(s: dict) -> None:
    l1 = s["layer1_citation_quality"]
    l2 = s["layer2_accuracy"]
    k  = K

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions     : {s['total_questions']}")
    print(f"Questions with GT   : {s['questions_with_gt']}")
    print()
    print("── Layer 1: Citation Quality (Retrieval) ─────────────────")
    print(f"  Section Hit Rate @{k}  : {l1[f'section_hit_rate@{k}']:.1%}"
          f"  ({l1['n_evaluated']} questions)")
    print(f"  MRR                   : {l1['MRR']:.3f}")
    print(f"  * {l1['note']}")
    print()
    print("── Layer 2: Accuracy (Generation) ───────────────────────")
    print(f"  Escalation accuracy   : {l2['escalation_accuracy']:.1%}"
          f"  ({l2['escalation_correct']}/{l2['escalation_total']} "
          f"not_found correctly refused)")
    print(f"  Confidence match      : {l2['confidence_match_rate']:.1%}")
    print(f"  Citation coverage     : {l2['citation_coverage']:.1%}")
    print()
    print("  Confidence distribution:")
    for conf, cnt in sorted(l2["confidence_distribution"].items(),
                            key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"    {conf:<12} {cnt:3d}  {bar}")
    print()
    print("  Manual review:")
    print(f"    Answer correctness   : {l2['human_answer_correctness']}")
    print(f"    Citation accuracy    : {l2['human_citation_accuracy']}")
    print(f"    Reviewed             : {l2['manual_reviewed']} questions")
    if l2["manual_reviewed"] == 0:
        print()
        print("  → Open evaluation/results.json and fill in:")
        print("    'human_correct': true/false   (answer clinically correct?)")
        print("    'citation_correct': true/false (citation supports answer?)")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets",    default="hand_crafted,transcript")
    parser.add_argument("--output",  default="evaluation/results.json")
    parser.add_argument("--summary", default="evaluation/summary.json")
    parser.add_argument("--delay",   type=float, default=1.0)
    args = parser.parse_args()

    run_evaluation(
        sets         = [s.strip() for s in args.sets.split(",")],
        output_path  = args.output,
        summary_path = args.summary,
        delay        = args.delay,
    )
