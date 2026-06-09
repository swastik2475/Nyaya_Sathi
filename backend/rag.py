"""
rag.py — NyayaSathi RAG pipeline.

Features:
  - Hybrid retrieval: FAISS (dense) + BM25 (sparse) with score fusion
  - HyDE query expansion for better semantic recall
  - Pluggable LLM: Groq (default) → OpenRouter → Ollama fallback
  - Source citation extraction from chunk metadata
  - Confidence score based on retrieval distances
"""

import os
import pickle
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    EMBED_MODEL, FAISS_PATH, BM25_PATH,
    TOP_K, HYBRID_ALPHA, MAX_HISTORY_TURNS,
)

# ─── Embeddings ───────────────────────────────────────────────────────────────

_embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ─── Vector store ─────────────────────────────────────────────────────────────

_db: Optional[FAISS] = None

def _load_faiss() -> FAISS:
    global _db
    if _db is None:
        if not Path(FAISS_PATH).exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{FAISS_PATH}'. "
                "Run `python ingestion.py` first to build the index."
            )
        _db = FAISS.load_local(
            FAISS_PATH,
            _embeddings,
            allow_dangerous_deserialization=True,
        )
    return _db


def is_index_loaded() -> bool:
    try:
        _load_faiss()
        return True
    except FileNotFoundError:
        return False

# ─── BM25 (sparse retrieval) ──────────────────────────────────────────────────

_bm25 = None
_bm25_docs: list[Document] = []

def _load_bm25():
    global _bm25, _bm25_docs
    if _bm25 is not None:
        return _bm25

    bm25_file = Path(BM25_PATH) / "bm25.pkl"
    if not bm25_file.exists():
        return None  # gracefully fall back to dense-only

    try:
        from rank_bm25 import BM25Okapi
        with open(bm25_file, "rb") as f:
            data = pickle.load(f)
        _bm25 = data["bm25"]
        _bm25_docs = data["docs"]
        return _bm25
    except Exception:
        return None


def _bm25_search(query: str, k: int) -> list[Document]:
    bm25 = _load_bm25()
    if bm25 is None or not _bm25_docs:
        return []

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [_bm25_docs[i] for i in top_indices]


# ─── LLM ──────────────────────────────────────────────────────────────────────

_llm = None

def _get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=1024,
        )

    elif LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            model=OPENROUTER_MODEL,
            temperature=0.2,
            max_tokens=1024,
            default_headers={
                "HTTP-Referer": "https://nyayasathi.app",
                "X-Title": "NyayaSathi",
            },
        )

    else:  # ollama
        from langchain_ollama import ChatOllama
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )

    return _llm


# ─── HyDE Query Expansion ─────────────────────────────────────────────────────

_HYDE_PROMPT = """You are a legal document author. Write a short hypothetical passage (3-4 sentences) 
from an Indian legal document that would answer this question. Be factual and cite relevant laws.
Do NOT answer the question — write as if you are the source document.

Question: {question}
Hypothetical legal text:"""

def _hyde_expand(question: str) -> str:
    """Generate a hypothetical document for better semantic retrieval."""
    try:
        llm = _get_llm()
        resp = llm.invoke(_HYDE_PROMPT.format(question=question))
        return resp.content.strip()
    except Exception:
        return question  # fall back to original query on error


# ─── Hybrid Retrieval ─────────────────────────────────────────────────────────

def _hybrid_retrieve(question: str, k: int = TOP_K) -> tuple[list[Document], float]:
    """
    Combine FAISS dense + BM25 sparse retrieval via reciprocal rank fusion.
    Returns (merged_docs, avg_confidence).
    """
    db = _load_faiss()

    # Dense retrieval with HyDE-expanded query
    hyde_query   = _hyde_expand(question)
    dense_results = db.similarity_search_with_score(hyde_query, k=k)
    dense_docs    = [doc for doc, _ in dense_results]
    dense_scores  = [score for _, score in dense_results]

    # BM25 sparse retrieval on original query
    sparse_docs = _bm25_search(question, k=k)

    # Reciprocal rank fusion
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:100]
        doc_scores[key] = doc_scores.get(key, 0) + HYBRID_ALPHA * (1 / (rank + 1))
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content[:100]
        doc_scores[key] = doc_scores.get(key, 0) + (1 - HYBRID_ALPHA) * (1 / (rank + 1))
        doc_map[key] = doc

    sorted_keys = sorted(doc_scores, key=doc_scores.__getitem__, reverse=True)[:k]
    merged = [doc_map[k] for k in sorted_keys]

    # Confidence: inverse of avg distance (FAISS uses L2, lower = better)
    confidence = 0.0
    if dense_scores:
        avg_dist   = sum(dense_scores[:3]) / min(3, len(dense_scores))
        confidence = max(0.0, min(1.0, 1.0 - avg_dist / 2.0))

    return merged, confidence


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are NyayaSathi, a knowledgeable AI legal assistant specialising in Indian law.

Rules:
1. Answer ONLY from the provided Legal Context below.
2. If the answer is not in the context, say: "I don't have enough legal information on this topic. Please consult a qualified advocate."
3. Cite sources inline, e.g. (IPC §302) or (Article 21, Constitution).
4. Use simple language. Avoid Latin/jargon unless you define it.
5. If the question involves an emergency or serious crime, also mention: "Dial 112 (Police) or 1516 (Legal Aid)."
6. Never fabricate laws, sections, or case names.
7. Keep answers concise — under 350 words unless complexity demands more."""


# ─── Main Interface ───────────────────────────────────────────────────────────

def ask_rag(question: str, history: list[dict] | None = None) -> dict:
    """
    Full RAG pipeline entry point.

    Args:
        question: User's legal question
        history:  List of {"role": "user"|"assistant", "content": str}

    Returns:
        {
            "answer":         str,
            "sources":        list[str],   # friendly source names
            "source_details": list[dict],  # {title, snippet}
            "confidence":     float,
        }
    """
    history = history or []

    # 1. Retrieve
    docs, confidence = _hybrid_retrieve(question)

    # 2. Build context string
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # 3. Build history string (last N turns)
    history_text = ""
    if history:
        pairs = []
        for msg in history[-MAX_HISTORY_TURNS * 2:]:
            pairs.append(f"{msg['role'].capitalize()}: {msg['content']}")
        history_text = "\nConversation history:\n" + "\n".join(pairs) + "\n"

    # 4. Full prompt
    prompt = f"""{SYSTEM_PROMPT}
{history_text}
Legal Context:
{context}

User Question: {question}

Answer (cite sources inline):"""

    # 5. LLM call
    llm    = _get_llm()
    answer = llm.invoke(prompt).content.strip()

    # 6. Extract unique sources
    sources = list({doc.metadata.get("source", "Unknown") for doc in docs})

    source_details = [
        {
            "title":   doc.metadata.get("source", "Unknown"),
            "snippet": doc.page_content[:200].replace("\n", " "),
        }
        for doc in docs[:3]
    ]

    return {
        "answer":         answer,
        "sources":        sources,
        "source_details": source_details,
        "confidence":     round(confidence, 3),
    }


# Import here to avoid circular at module level
