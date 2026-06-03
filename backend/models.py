"""
models.py — Pydantic schemas for all API request/response bodies.
Keep everything typed; FastAPI validates automatically.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ─── Inbound ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="UUID identifying this chat session",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's legal question in English or Hindi",
        examples=["What are my rights if police arrests me without a warrant?"],
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message_id: str = Field(..., description="ID of the AI message being rated")
    rating: int     = Field(..., ge=0, le=1, description="1 = helpful, 0 = not helpful")
    comment: Optional[str] = Field(None, max_length=500)


# ─── Outbound ─────────────────────────────────────────────────────────────────

class Source(BaseModel):
    title: str  = Field(..., description="Friendly name, e.g. 'IPC Section 302'")
    snippet: str = Field(..., description="Relevant excerpt from the source document")


class QueryResponse(BaseModel):
    answer: str            = Field(..., description="AI-generated legal answer")
    sources: list[str]     = Field(default_factory=list, description="Source titles cited")
    source_details: list[Source] = Field(default_factory=list)
    message_id: str        = Field(..., description="Unique ID for this response (for feedback)")
    confidence: float      = Field(default=0.0, ge=0.0, le=1.0, description="Retrieval confidence score")


class HistoryMessage(BaseModel):
    role: str    = Field(..., description="'user' or 'assistant'")
    content: str = Field(...)
    ts: Optional[str] = Field(None, description="ISO timestamp")


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_model: str
    index_loaded: bool