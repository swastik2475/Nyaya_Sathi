"""
main.py — NyayaSathi FastAPI application.

Routes:
  GET  /health              → liveness + readiness check
  POST /query               → main RAG query endpoint
  GET  /history/{session}   → fetch chat history for a session
  DELETE /history/{session} → clear a session
  POST /feedback            → store 👍/👎 ratings

Run:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import uuid
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import validate_config, LLM_PROVIDER, GROQ_MODEL, OPENROUTER_MODEL, OLLAMA_MODEL
from models import (
    QueryRequest, QueryResponse,
    FeedbackRequest,
    HistoryResponse, HistoryMessage,
    HealthResponse,
)
from memory import init_db, get_history, save_turn, save_feedback, clear_session
from rag import ask_rag, is_index_loaded

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("nyayasathi")

# ─── Startup ──────────────────────────────────────────────────────────────────

validate_config()
init_db()

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NyayaSathi API",
    description="Indian Legal AI Assistant — RAG-powered backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Error handler ────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Liveness + readiness probe."""
    model_name = {
        "groq":       GROQ_MODEL,
        "openrouter": OPENROUTER_MODEL,
        "ollama":     OLLAMA_MODEL,
    }.get(LLM_PROVIDER, "unknown")

    return HealthResponse(
        status="ok",
        llm_provider=LLM_PROVIDER,
        llm_model=model_name,
        index_loaded=is_index_loaded(),
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(req: QueryRequest):
    """
    Main RAG endpoint.
    Accepts a question + session_id, returns an answer with cited sources.
    """
    log.info("Query session=%s | q=%s", req.session_id[:8], req.question[:80])

    try:
        history = get_history(req.session_id)
        result  = ask_rag(req.question, history)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error("RAG pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="RAG pipeline failed. Check backend logs.")

    message_id = save_turn(
        session_id=req.session_id,
        question=req.question,
        answer=result["answer"],
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        source_details=result["source_details"],
        message_id=message_id,
        confidence=result["confidence"],
    )


@app.get("/history/{session_id}", response_model=HistoryResponse, tags=["Memory"])
def history(session_id: str):
    """Fetch conversation history for a session."""
    msgs = get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryMessage(**m) for m in msgs],
    )


@app.delete("/history/{session_id}", tags=["Memory"])
def delete_history(session_id: str):
    """Clear all messages for a session (new chat)."""
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/feedback", tags=["Feedback"])
def feedback(req: FeedbackRequest):
    """Store 👍 (rating=1) or 👎 (rating=0) for an AI response."""
    save_feedback(
        session_id=req.session_id,
        message_id=req.message_id,
        rating=req.rating,
        comment=req.comment,
    )
    log.info(
        "Feedback session=%s | msg=%s | rating=%d",
        req.session_id[:8], req.message_id[:8], req.rating,
    )
    return {"status": "recorded"}