"""
generator.py — LLM generation with retrieved context.
Supports LLM_PROVIDER=openai (default) and LLM_PROVIDER=azure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.azure_config import get_openai_client, get_chat_model

# ── System prompt (authoritative from SKILL.md — do not soften) ──────────────

SYSTEM_PROMPT = """You are an immunisation guidelines assistant supporting clinical advisors at IMAC \
(Immunisation Advisory Centre, University of Auckland).

Rules you must follow without exception:
1. Answer ONLY using the provided reference sections below.
2. Every claim in your answer must cite its source (document name, chapter/section, URL).
3. If the answer is not clearly present in the references, respond:
   "I could not find a clear answer in the approved guidance. Please consult the relevant \
handbook section directly or escalate to a senior advisor."
4. Never speculate, infer beyond what is written, or fabricate clinical information.
5. Do not make diagnostic or treatment recommendations.
6. Keep answers concise and structured for a clinical professional.
7. Always remind the advisor that final clinical decisions remain with qualified staff.

You must return a JSON object with this exact structure:
{
  "answer": "<your answer here>",
  "citations": [
    {
      "source": "<document name>",
      "section": "<chapter/section title>",
      "url": "<url or empty string>",
      "excerpt": "<brief verbatim or near-verbatim quote supporting this claim>"
    }
  ],
  "confidence": "<high | medium | low | not_found>"
}

Confidence guidelines:
  - high     : answer is explicitly and clearly stated in the retrieved sections
  - medium   : answer can be reasonably inferred from the retrieved sections
  - low      : retrieved sections are only tangentially relevant
  - not_found: no relevant information found in the retrieved sections
"""


def _format_chunks_for_prompt(chunks: list[dict]) -> str:
    if not chunks:
        return "No reference sections were retrieved."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        breadcrumb = chunk.get("breadcrumb") or chunk.get("source_name", "Unknown source")
        url = chunk.get("url", "")
        content = chunk.get("content", "")
        parts.append(f"[{i}] {breadcrumb}\nURL: {url}\n{content}")
    return "\n\n---\n\n".join(parts)


def generate(query: str, chunks: list[dict]) -> dict:
    """
    Generate a structured response given a query and retrieved chunks.
    Works with both LLM_PROVIDER=openai and LLM_PROVIDER=azure.
    """
    context = _format_chunks_for_prompt(chunks)
    user_message = (
        f"Reference sections:\n{context}\n\n"
        f"Advisor query: {query}"
    )

    client = get_openai_client()
    model = get_chat_model()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=1500,
    )

    raw = response.choices[0].message.content or "{}"

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw output:\n{raw}") from e

    result.setdefault("answer", "")
    result.setdefault("citations", [])
    result.setdefault("confidence", "not_found")
    return result
