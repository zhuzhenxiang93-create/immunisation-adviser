"""
run_eval.py — Run the agent against the curated question set and save results.

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --output evaluation/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.query_handler import run_query

QUESTION_SET_PATH = Path(__file__).parent / "question_set.json"


def run_evaluation(output_path: str | None = None) -> list[dict]:
    with open(QUESTION_SET_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    print(f"Running evaluation on {len(questions)} questions...\n")

    for q in questions:
        qid = q["id"]
        query = q["query"]
        print(f"[{qid}] {query[:80]}...")

        result = run_query(query)
        output = result.get("output", {})

        entry = {
            "id": qid,
            "category": q.get("category", ""),
            "query": query,
            "answer": output.get("answer", ""),
            "confidence": output.get("confidence", "not_found"),
            "citations": output.get("citations", []),
            "chunks_retrieved": output.get("audit", {}).get("chunks_retrieved", 0),
            "expected_confidence": q.get("expected_confidence", ""),
            "ground_truth_answer": q.get("ground_truth_answer", ""),
            "ground_truth_citation": q.get("ground_truth_citation", {}),
            # Manual review fields — fill in after running
            "human_correct": None,
            "citation_correct": None,
            "notes": "",
        }
        results.append(entry)
        print(f"  → confidence={entry['confidence']}, chunks={entry['chunks_retrieved']}\n")

        # Avoid rate limits
        time.sleep(1)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {out}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()
    run_evaluation(args.output)
