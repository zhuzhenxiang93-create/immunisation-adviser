"""
query_handler.py — Main agent entry point using LangGraph.

Graph structure:
    START → retrieve_node → generate_node → format_node → END

State carries the query, retrieved chunks, raw generation, and final output
through the pipeline. Each node is a pure function with no side effects.

Usage:
    from agent.query_handler import run_query
    result = run_query("When should MMR vaccine be given to a 12-month-old?")
    print(result["formatted"])
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.retriever import retrieve
from agent.generator import generate
from agent.output_formatter import build_output, format_for_display

from config.azure_config import RETRIEVAL_TOP_K


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    chunks: list[dict]          # retrieved from Azure AI Search
    generation: dict            # raw LLM output (answer + citations + confidence)
    output: dict                # final structured output including audit
    formatted: str              # human-readable string for UI display
    error: str                  # non-empty if any node failed


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant chunks from Azure AI Search (hybrid search)."""
    try:
        chunks = retrieve(state["query"], top_k=RETRIEVAL_TOP_K)
        print(f"[retrieve] {len(chunks)} chunks retrieved")
        return {"chunks": chunks, "error": ""}
    except Exception as e:
        print(f"[retrieve] ERROR: {e}")
        return {"chunks": [], "error": f"Retrieval failed: {e}"}


def generate_node(state: AgentState) -> dict:
    """Generate answer + citations from LLM given retrieved chunks."""
    if state.get("error"):
        # Propagate retrieval error — skip generation
        return {"generation": {"answer": "", "citations": [], "confidence": "not_found"}}
    try:
        generation = generate(state["query"], state["chunks"])
        print(f"[generate] confidence={generation.get('confidence')}")
        return {"generation": generation, "error": ""}
    except Exception as e:
        print(f"[generate] ERROR: {e}")
        return {
            "generation": {"answer": "", "citations": [], "confidence": "not_found"},
            "error": f"Generation failed: {e}",
        }


def format_node(state: AgentState) -> dict:
    """Build final structured output and human-readable display string."""
    output = build_output(
        query=state["query"],
        generation_result=state["generation"],
        chunks_retrieved=len(state.get("chunks", [])),
    )

    # Inject error message into answer if something went wrong
    if state.get("error"):
        output["answer"] = (
            f"An error occurred: {state['error']}\n\n"
            "Please try again or contact technical support."
        )
        output["confidence"] = "not_found"

    formatted = format_for_display(output)
    return {"output": output, "formatted": formatted}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("generate_node", generate_node)
    builder.add_node("format_node", format_node)

    builder.add_edge(START, "retrieve_node")
    builder.add_edge("retrieve_node", "generate_node")
    builder.add_edge("generate_node", "format_node")
    builder.add_edge("format_node", END)

    return builder.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_query(query: str) -> dict:
    """
    Run the full RAG pipeline for a single query.

    Args:
        query: Free-text question from a clinical advisor.

    Returns:
        AgentState dict with keys:
            query, chunks, generation, output, formatted, error
    """
    graph = _get_graph()
    initial_state: AgentState = {
        "query": query,
        "chunks": [],
        "generation": {},
        "output": {},
        "formatted": "",
        "error": "",
    }
    return graph.invoke(initial_state)


# ---------------------------------------------------------------------------
# CLI — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    test_query = (
        input("Enter a query (or press Enter for default): ").strip()
        or "When should the MMR vaccine be given to a 12-month-old child in New Zealand?"
    )

    print(f"\nQuery: {test_query}\n{'=' * 60}")
    result = run_query(test_query)

    print("\n--- Formatted Output ---")
    print(result["formatted"])

    print("\n--- Structured JSON ---")
    print(json.dumps(result["output"], indent=2, ensure_ascii=False))
