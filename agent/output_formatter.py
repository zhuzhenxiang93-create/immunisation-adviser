"""
output_formatter.py — Build the final structured output and audit log.

Final output schema (from SKILL.md):
{
  "answer": "...",
  "citations": [...],
  "confidence": "high | medium | low | not_found",
  "audit": {
    "query": "...",
    "chunks_retrieved": 5,
    "timestamp": "..."
  }
}
"""

from __future__ import annotations

from datetime import datetime, timezone


def build_output(
    query: str,
    generation_result: dict,
    chunks_retrieved: int,
) -> dict:
    """
    Merge LLM generation result with audit metadata.
    Returns the complete output dict.
    """
    return {
        "answer": generation_result.get("answer", ""),
        "citations": generation_result.get("citations", []),
        "confidence": generation_result.get("confidence", "not_found"),
        "audit": {
            "query": query,
            "chunks_retrieved": chunks_retrieved,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


def format_for_display(output: dict) -> str:
    """
    Render the structured output as readable text for the Gradio UI.
    """
    lines = []

    confidence = output.get("confidence", "not_found")
    confidence_label = {
        "high": "HIGH",
        "medium": "MEDIUM",
        "low": "LOW (please verify manually)",
        "not_found": "NOT FOUND",
    }.get(confidence, confidence.upper())

    lines.append(f"**Confidence:** {confidence_label}\n")
    lines.append("---\n")
    lines.append(output.get("answer", "No answer generated."))
    lines.append("")

    citations = output.get("citations", [])
    if citations:
        lines.append("\n**Sources:**")
        for i, c in enumerate(citations, 1):
            source = c.get("source", "")
            section = c.get("section", "")
            url = c.get("url", "")
            excerpt = c.get("excerpt", "")

            ref_line = f"{i}. {source}"
            if section:
                ref_line += f" — {section}"
            if url:
                ref_line += f"  \n   [{url}]({url})"
            if excerpt:
                ref_line += f"  \n   > _{excerpt}_"
            lines.append(ref_line)

    lines.append("\n---")
    lines.append(
        "_This information is provided to support clinical advisors. "
        "Final clinical decisions remain with qualified staff._"
    )

    return "\n".join(lines)


