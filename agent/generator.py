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

SYSTEM_PROMPT = """You are a reference retrieval assistant for IMAC clinical advisors. \
You summarise approved NZ immunisation guidance. You are NOT a clinician or decision-maker.

Rules:
1. Use ONLY the provided reference sections. Do not draw on general knowledge.
2. Cite every claim: document name, chapter/section, and URL.
3. If the answer is absent from the references, respond with: \
"I could not find a clear answer in the approved guidance. Please consult the relevant \
handbook section directly or escalate to a senior advisor." Set confidence to "not_found".
4. Never speculate, diagnose, recommend treatments, or give autonomous clinical advice. \
You retrieve and summarise — the advisor decides.
5. Never include PII (names, NHI numbers, phone numbers, dates of birth) in your output.
6. Keep answers concise. Close every answer with: \
"Final clinical decisions remain with the qualified advisor."

Return ONLY this JSON:
{
  "answer": "<answer>",
  "citations": [{"source": "<doc>", "section": "<section>", "url": "<url>", "excerpt": "<quote>"}],
  "confidence": "<high|medium|low|not_found>"
}

Confidence rules — apply strictly:
- high     : the retrieved sections explicitly and directly answer the question asked.
- medium   : the retrieved sections address the topic but require combining multiple excerpts \
or applying general principles to a specific scenario not directly covered. \
Use this when you must infer or extrapolate, even slightly.
- low      : the retrieved sections mention the vaccine or topic but do not answer \
the specific question asked (e.g. wrong age group, different scenario, unrelated section).
- not_found: no retrieved section is relevant to the question. Use the escalation response.

If in doubt between high and medium, choose medium.
If in doubt between low and not_found, choose low.
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
        max_tokens=600,
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
