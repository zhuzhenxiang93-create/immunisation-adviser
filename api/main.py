"""
api/main.py — FastAPI backend for the Immunisation Guidelines Adviser Agent.

Endpoints:
  GET  /health              Health check
  POST /auth/register       Register a new user
  POST /auth/login          Login → JWT token
  GET  /auth/me             Get current user info
  POST /query               Run a single query (auth required)
  POST /query/stream        Stream the answer token-by-token / SSE (auth required)
  GET  /history             Last N queries in this session (in-memory)
  DELETE /history           Clear session history

Run:
  conda activate immunisation-adviser
  cd D:/714/hackthon/immunisation-adviser
  uvicorn api.main:app --reload --port 8000

Docs:  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
import asyncio
from collections import deque
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


# ── In-memory session history (last 50 queries) ───────────────────────────────
_history: deque[dict] = deque(maxlen=50)


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


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    confidence: str          # high | medium | low | not_found
    audit: AuditModel
    formatted: str           # human-readable markdown


class HistoryEntry(BaseModel):
    query: str
    confidence: str
    chunks_retrieved: int
    timestamp: str


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


# ── Query endpoints (auth required) ───────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["adviser"])
def query(req: QueryRequest, current_user: dict = Depends(get_current_user)):
    """
    Run a clinical query through the full RAG pipeline.
    Returns a structured answer with citations and confidence indicator.
    """
    try:
        result = run_query(req.query)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    output = result.get("output", {})
    audit = output.get("audit", {})

    _history.appendleft({
        "query": req.query,
        "confidence": output.get("confidence", "not_found"),
        "chunks_retrieved": audit.get("chunks_retrieved", 0),
        "timestamp": audit.get("timestamp", ""),
    })

    return QueryResponse(
        answer=output.get("answer", ""),
        citations=[CitationModel(**c) for c in output.get("citations", [])],
        confidence=output.get("confidence", "not_found"),
        audit=AuditModel(
            query=audit.get("query", req.query),
            chunks_retrieved=audit.get("chunks_retrieved", 0),
            timestamp=audit.get("timestamp", ""),
        ),
        formatted=result.get("formatted", ""),
    )


@app.post("/query/stream", tags=["adviser"])
async def query_stream(req: QueryRequest, current_user: dict = Depends(get_current_user)):
    """
    Stream the answer token-by-token using Server-Sent Events (SSE).
    """

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

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {delta}\n\n"

            audit_payload = (
                f'{{"chunks_retrieved": {len(chunks)}, '
                f'"timestamp": "{datetime.now(timezone.utc).isoformat()}"}}'
            )
            yield f"event: audit\ndata: {audit_payload}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

            _history.appendleft({
                "query": req.query,
                "confidence": "streamed",
                "chunks_retrieved": len(chunks),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        except Exception as e:
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Session history ────────────────────────────────────────────────────────────

@app.get("/history", response_model=list[HistoryEntry], tags=["session"])
def get_history(limit: int = 10, current_user: dict = Depends(get_current_user)):
    """Return the last N queries in this session (in-memory, not persisted)."""
    return list(_history)[:limit]


@app.delete("/history", tags=["session"])
def clear_history(current_user: dict = Depends(get_current_user)):
    """Clear session history."""
    _history.clear()
    return {"status": "cleared"}
