"""
api/main.py — FastAPI backend for the Immunisation Guidelines Adviser Agent.

Endpoints:
  GET  /health              Health check
  POST /auth/register       Register a new user
  POST /auth/login          Login → JWT token
  GET  /auth/me             Get current user info
  POST /query               Run a single query (auth required)
  POST /query/stream        Stream the answer token-by-token / SSE (auth required)
  GET  /history             Last N queries (persisted in SQLite)
  DELETE /history           Clear query history
  GET  /reports/summary     Aggregate query statistics

Run:
  conda activate immunisation-adviser
  cd D:/714/hackthon/immunisation-adviser
  uvicorn api.main:app --reload --port 8000

Docs:  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.query_handler import run_query
from agent.generator import SYSTEM_PROMPT, _format_chunks_for_prompt
from agent.retriever import retrieve
from config.azure_config import get_openai_client, get_chat_model
from api.auth_manager import AuthManager
from api.jwt_utils import create_access_token, verify_token
from api.audit_logger import AuditLogger
from api.pii_filter import scan as pii_scan, redact_output

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IMAC Immunisation Guidelines Adviser",
    description=(
        "RAG-based agent that retrieves answers from approved NZ immunisation guidance. "
        "For clinical advisor support only — final decisions remain with qualified staff."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the Bootstrap UI at /
_static_dir = Path(__file__).parent.parent / "ui" / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(str(_static_dir / "index.html"))

# ── Auth setup ────────────────────────────────────────────────────────────────

_auth = AuthManager(db_path="./data/users.db")
_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """Dependency: require a valid JWT. Raises 401 if missing/invalid."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id = verify_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = _auth.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ── Persistent audit log (SQLite) ────────────────────────────────────────────
_audit = AuditLogger(db_path="./data/users.db")


# ── Request / Response schemas ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=32)
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str
    email: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=20)


class CitationModel(BaseModel):
    source: str
    section: str
    url: str
    excerpt: str


class AuditModel(BaseModel):
    query: str
    chunks_retrieved: int
    timestamp: str


class ClassificationModel(BaseModel):
    vaccine_type:      list[str] = []
    query_type:        list[str] = []
    clinical_scenario: list[str] = []
    caller_type:       str = "unknown"
    patient_age_group: str = "unknown"
    urgency:           str = "routine"


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    confidence: str          # high | medium | low | not_found
    classification: ClassificationModel
    audit: AuditModel
    formatted: str           # human-readable markdown


class HistoryEntry(BaseModel):
    query: str
    confidence: str
    chunks_retrieved: int
    timestamp: str


class ReportSummary(BaseModel):
    total_queries:     int
    confidence:        dict
    vaccine_type:      dict
    query_type:        dict
    clinical_scenario: dict
    urgency:           dict
    patient_age_group: dict
    daily_volume:      dict


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=AuthResponse, tags=["auth"])
def auth_register(req: RegisterRequest):
    """Register a new account and return a JWT token (auto-login)."""
    try:
        user = _auth.register_user(req.username, req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_access_token(user["id"], user["username"])
    return AuthResponse(token=token, username=user["username"], email=user["email"])


@app.post("/auth/login", response_model=AuthResponse, tags=["auth"])
def auth_login(req: LoginRequest):
    """Login with username + password and return a JWT token."""
    user = _auth.authenticate_user(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(user["id"], user["username"])
    return AuthResponse(token=token, username=user["username"], email=user["email"])


@app.get("/auth/me", tags=["auth"])
def auth_me(current_user: dict = Depends(get_current_user)):
    """Return the current user's profile."""
    return current_user


# ── System endpoints ───────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "IMAC Immunisation Adviser API",
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

@app.get("/reports/summary", response_model=ReportSummary, tags=["reporting"])
def reports_summary(current_user: dict = Depends(get_current_user)):
    """
    Aggregate statistics across all logged queries.
    Returns distribution by confidence, vaccine type, query type,
    clinical scenario, urgency, patient age group, and daily volume.
    """
    return _audit.get_summary()


# ── Query endpoints (auth required) ───────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["adviser"])
def query(req: QueryRequest, current_user: dict = Depends(get_current_user)):
    """
    Run a clinical query through the full RAG pipeline.
    Returns a structured answer with citations and confidence indicator.
    """
    pii = pii_scan(req.query)
    if pii["has_pii"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Query contains potential personal information ({', '.join(pii['types'])}). "
                "Please remove patient identifiers before submitting."
            ),
        )

    try:
        result = run_query(req.query)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    output = result.get("output", {})
    audit_meta = output.get("audit", {})

    clf = output.get("classification", {})
    confidence = output.get("confidence", "not_found")
    raw_citations = output.get("citations", [])
    safe_answer = redact_output(output.get("answer", ""))

    # ── Responsible AI enforcement ────────────────────────────────────────────
    # Accuracy over recall: when the model signals no answer, enforce the
    # escalation message regardless of what the LLM actually returned.
    _ESCALATION = (
        "I could not find a clear answer in the approved guidance. "
        "Please consult the relevant handbook section directly or escalate "
        "to a senior advisor.\n\n"
        "Final clinical decisions remain with the qualified advisor."
    )
    if confidence == "not_found":
        safe_answer = _ESCALATION
        raw_citations = []
    elif confidence == "low":
        safe_answer = (
            safe_answer
            + "\n\n⚠ Low confidence: the retrieved guidance is only tangentially relevant. "
            "Please verify with the primary source before acting on this information."
        )

    # Source transparency: warn if answer was generated but no citations provided
    if confidence not in ("not_found",) and not raw_citations:
        safe_answer = (
            safe_answer
            + "\n\n⚠ No source citations were returned. Treat this answer with caution "
            "and verify directly against the approved handbook."
        )

    chunks_retrieved = audit_meta.get("chunks_retrieved", 0)
    sources = result.get("chunks", [])  # raw chunks list if pipeline exposes it

    _audit.log(
        query=req.query,
        confidence=confidence,
        chunks_retrieved=chunks_retrieved,
        classification=clf,
        username=current_user.get("username", "unknown"),
        sources_retrieved=sources,
        citations=raw_citations,
        answer=safe_answer,
    )
    return QueryResponse(
        answer=safe_answer,
        citations=[CitationModel(**c) for c in raw_citations],
        confidence=confidence,
        classification=ClassificationModel(
            vaccine_type=clf.get("vaccine_type", []),
            query_type=clf.get("query_type", []),
            clinical_scenario=clf.get("clinical_scenario", []),
            caller_type=clf.get("caller_type", "unknown"),
            patient_age_group=clf.get("patient_age_group", "unknown"),
            urgency=clf.get("urgency", "routine"),
        ),
        audit=AuditModel(
            query=audit_meta.get("query", req.query),
            chunks_retrieved=chunks_retrieved,
            timestamp=audit_meta.get("timestamp", ""),
        ),
        formatted=result.get("formatted", ""),
    )


@app.post("/query/stream", tags=["adviser"])
async def query_stream(req: QueryRequest, current_user: dict = Depends(get_current_user)):
    """
    Stream the answer token-by-token using Server-Sent Events (SSE).
    """
    # PII check on input (same guard as /query)
    pii = pii_scan(req.query)
    if pii["has_pii"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Query contains potential personal information ({', '.join(pii['types'])}). "
                "Please remove patient identifiers before submitting."
            ),
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, lambda: retrieve(req.query, req.top_k)
            )

            context = _format_chunks_for_prompt(chunks)
            user_message = (
                f"Reference sections:\n{context}\n\n"
                f"Advisor query: {req.query}"
            )

            client = get_openai_client()
            model = get_chat_model()

            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                stream=True,
                temperature=0,
                max_tokens=1500,
            )

            # Accumulate full response so we can redact PII before streaming
            full_response: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_response.append(delta)

            safe_text = redact_output("".join(full_response))
            yield f"data: {safe_text}\n\n"

            audit_payload = (
                f'{{"chunks_retrieved": {len(chunks)}, '
                f'"timestamp": "{datetime.now(timezone.utc).isoformat()}"}}'
            )
            yield f"event: audit\ndata: {audit_payload}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

            _audit.log(
                query=req.query,
                confidence="streamed",
                chunks_retrieved=len(chunks),
                classification={},
                username=current_user.get("username", "unknown"),
            )

        except Exception as e:
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Query history (persisted) ─────────────────────────────────────────────────

@app.get("/history", response_model=list[HistoryEntry], tags=["session"])
def get_history(limit: int = 20, current_user: dict = Depends(get_current_user)):
    """Return the last N queries for the current user (persisted in SQLite)."""
    rows = _audit.get_recent(limit=limit, username=current_user.get("username"))
    return [
        HistoryEntry(
            query=r["query"],
            confidence=r["confidence"],
            chunks_retrieved=r["chunks_retrieved"],
            timestamp=r["timestamp"],
        )
        for r in rows
    ]


@app.delete("/history", tags=["session"])
def clear_history(current_user: dict = Depends(get_current_user)):
    """Clear query history for the current user."""
    _audit.clear(username=current_user.get("username"))
    return {"status": "cleared"}


