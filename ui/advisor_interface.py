"""
advisor_interface.py — Gradio demo UI for the Immunisation Guidelines Adviser Agent.

Run:
    python -m ui.advisor_interface
    # or
    python immunisation-adviser/ui/advisor_interface.py

Opens a local web UI at http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.query_handler import run_query


# ---------------------------------------------------------------------------
# Example queries pre-loaded in the UI
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "When should the MMR vaccine be given to a 12-month-old child in New Zealand?",
    "Is the influenza vaccine safe for a patient with egg allergy?",
    "What is the catch-up schedule for a 3-year-old who has never been vaccinated?",
    "How should varicella vaccine be stored?",
    "What are the contraindications for the BCG vaccine?",
]


# ---------------------------------------------------------------------------
# Core handler
# ---------------------------------------------------------------------------

def handle_query(query: str) -> tuple[str, str]:
    """
    Called by Gradio on submit.
    Returns (formatted_answer, raw_json_for_debug_tab).
    """
    if not query.strip():
        return "Please enter a question.", "{}"

    try:
        result = run_query(query.strip())
        formatted = result.get("formatted", "No output generated.")
        raw_json = json.dumps(result.get("output", {}), indent=2, ensure_ascii=False)
        return formatted, raw_json
    except Exception as e:
        error_msg = (
            f"**Error:** {e}\n\n"
            "_If Azure credentials are not yet configured, copy `.env.example` to `.env` "
            "and fill in your Azure keys._"
        )
        return error_msg, "{}"


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="IMAC Immunisation Guidelines Adviser") as demo:
        gr.Markdown(
            """
# IMAC Immunisation Guidelines Adviser
**Proof-of-concept RAG agent** — COMPSCI 714 AI Hackathon, University of Auckland

This tool supports clinical advisors by retrieving relevant sections from approved
NZ immunisation guidance and returning a structured answer with citations.

> ⚠️ **For advisor support only.** Final clinical decisions remain with qualified staff.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Advisor Query",
                    placeholder="e.g. When should the MMR vaccine be given to a 12-month-old?",
                    lines=3,
                )
                submit_btn = gr.Button("Submit", variant="primary")
                gr.Examples(
                    examples=EXAMPLE_QUERIES,
                    inputs=query_input,
                    label="Example queries",
                )

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Answer"):
                        answer_output = gr.Markdown(label="Response")
                    with gr.Tab("Raw JSON (debug)"):
                        json_output = gr.Code(language="json", label="Structured output")

        submit_btn.click(
            fn=handle_query,
            inputs=query_input,
            outputs=[answer_output, json_output],
        )
        query_input.submit(
            fn=handle_query,
            inputs=query_input,
            outputs=[answer_output, json_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False)
