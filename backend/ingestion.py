"""
ingestion.py — NyayaSathi data ingestion pipeline.

Run ONCE (or whenever the corpus changes):
    python ingestion.py

What it does:
    1. Loads local .txt files from data/corpus/
    2. Scrapes key legal websites (DOJ, NALSA, cybercrime.gov.in)
    3. Splits into chunks
    4. Builds FAISS (dense) index  →  data/faiss_index/
    5. Builds BM25  (sparse) index →  data/bm25_index/bm25.pkl
"""

import os
import sys
import pickle
import logging
from pathlib import Path

# Add parent dir to path so config imports work
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FAISS_PATH, BM25_PATH, CORPUS_DIR,
    EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingestion")

# ─── Scrape targets ───────────────────────────────────────────────────────────

WEB_SOURCES = [
    "https://doj.gov.in/about-department",
    "https://doj.gov.in/schemes",
    "https://doj.gov.in/tele-law",
    "https://doj.gov.in/fast-track-special-courts",
    "https://www.cybercrime.gov.in/",
    "https://nalsa.gov.in/",
    "https://nalsa.gov.in/lsams/legal-services",
]

# ─── Imports ──────────────────────────────────────────────────────────────────

def _check_deps():
    missing = []
    for pkg in ["langchain_community", "langchain_text_splitters",
                "faiss", "sentence_transformers", "rank_bm25", "bs4"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error("Missing packages: %s", missing)
        log.error("Run: pip install -r requirements.txt")
        sys.exit(1)

_check_deps()

from langchain_community.document_loaders import (
    WebBaseLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ─── 1. Load documents ────────────────────────────────────────────────────────

def load_local_corpus() -> list[Document]:
    corpus_path = Path(CORPUS_DIR)
    if not corpus_path.exists() or not any(corpus_path.glob("**/*.txt")):
        log.warning("No .txt files found in %s — skipping local corpus", CORPUS_DIR)
        return []

    log.info("📁 Loading local corpus from %s ...", CORPUS_DIR)
    loader = DirectoryLoader(
        CORPUS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    log.info("   → %d documents loaded", len(docs))
    return docs


def load_web_sources() -> list[Document]:
    log.info("🌐 Scraping %d legal websites ...", len(WEB_SOURCES))
    os.environ.setdefault("USER_AGENT", "NyayaSathi/1.0 (Legal AI; github.com/nyayasathi)")

    loaded = []
    for url in WEB_SOURCES:
        try:
            docs = WebBaseLoader(url).load()
            loaded.extend(docs)
            log.info("   ✓ %s (%d docs)", url, len(docs))
        except Exception as e:
            log.warning("   ✗ %s — %s", url, e)

    log.info("   → %d web documents total", len(loaded))
    return loaded


# ─── 2. Chunk ─────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[Document]) -> list[Document]:
    log.info("✂️  Splitting %d documents into chunks ...", len(docs))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "। ", ". ", " ", ""],  # Hindi-aware
    )
    chunks = splitter.split_documents(docs)
    log.info("   → %d chunks created", len(chunks))
    return chunks


# ─── 3. Build FAISS index ─────────────────────────────────────────────────────

def build_faiss(chunks: list[Document]) -> None:
    log.info("🔢 Building FAISS index (this may take a few minutes) ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    Path(FAISS_PATH).mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_PATH)
    log.info("   ✅ FAISS index saved → %s", FAISS_PATH)


# ─── 4. Build BM25 index ──────────────────────────────────────────────────────

def build_bm25(chunks: list[Document]) -> None:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.warning("rank_bm25 not installed — skipping BM25 index. pip install rank-bm25")
        return

    log.info("📊 Building BM25 sparse index ...")
    tokenised = [doc.page_content.lower().split() for doc in chunks]
    bm25      = BM25Okapi(tokenised)

    Path(BM25_PATH).mkdir(parents=True, exist_ok=True)
    out_path = Path(BM25_PATH) / "bm25.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": chunks}, f)

    log.info("   ✅ BM25 index saved → %s", out_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("═" * 60)
    log.info("  NyayaSathi Ingestion Pipeline")
    log.info("═" * 60)

    local_docs = load_local_corpus()
    web_docs   = load_web_sources()

    all_docs = local_docs + web_docs
    if not all_docs:
        log.error("No documents found. Add .txt files to data/corpus/ or check web URLs.")
        sys.exit(1)

    log.info("📝 Total source documents: %d", len(all_docs))
    chunks = chunk_documents(all_docs)

    build_faiss(chunks)
    build_bm25(chunks)

    log.info("═" * 60)
    log.info("✅  Ingestion complete. Start the API with:")
    log.info("   uvicorn main:app --reload --port 8000")
    log.info("═" * 60)


if __name__ == "__main__":
    main()