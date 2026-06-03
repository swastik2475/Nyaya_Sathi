"""
config.py — NyayaSathi central configuration
All env vars, model names, paths live here.
Import this everywhere instead of scattering os.getenv calls.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM Provider ─────────────────────────────────────────────────────────────
# Set LLM_PROVIDER in .env: "groq" | "openrouter" | "ollama"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()

# Groq  →  https://console.groq.com  (free, very fast)
GROQ_API_KEY: str  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str    = "llama-3.3-70b-versatile"          # 14 400 req/day free

# OpenRouter  →  https://openrouter.ai  (free models available)
OPENROUTER_API_KEY: str  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str    = "meta-llama/llama-3.1-8b-instruct:free"
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Ollama (fully local, no API key needed)
OLLAMA_MODEL: str    = "gemma2:2b"
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
FAISS_PATH  = os.path.join(DATA_DIR, "faiss_index")
BM25_PATH   = os.path.join(DATA_DIR, "bm25_index")
CORPUS_DIR  = os.path.join(DATA_DIR, "corpus")
DB_PATH     = os.path.join(DATA_DIR, "chat_history.db")

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K          = 5    # Number of chunks to retrieve
HYBRID_ALPHA   = 0.7  # Weight for dense (FAISS) vs sparse (BM25): 1.0 = pure dense

# ─── Memory ───────────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 6   # How many past turns to include in prompt context

# ─── Server ───────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ─── Validation ───────────────────────────────────────────────────────────────
def validate_config() -> None:
    """Call at startup to catch missing keys early."""
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        raise EnvironmentError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com"
        )
    if LLM_PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set. "
            "Get a free key at https://openrouter.ai"
        )
    if LLM_PROVIDER not in ("groq", "openrouter", "ollama"):
        raise ValueError(f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. Use groq | openrouter | ollama")