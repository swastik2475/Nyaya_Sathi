"""
evaluate_retrieval.py — NyayaSathi retrieval quality evaluation.

Tests dense (FAISS) and hybrid (FAISS + BM25) retrieval on 15 standard
Indian legal queries and reports Hit Rate @ TOP_K.

Usage:
    python evaluate_retrieval.py
"""

import sys
import json
import pickle
from pathlib import Path

# ── Ensure backend/ is importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import FAISS_PATH, BM25_PATH, EMBED_MODEL, TOP_K

# ── Evaluation queries with expected keyword signals ──────────────────────────
QUERIES = [
    {
        "query": "What are my rights if police arrests me without a warrant?",
        "keywords": ["arrest", "warrant", "rights", "grounds", "magistrate"],
    },
    {
        "query": "How do I file an FIR online?",
        "keywords": ["fir", "online", "cybercrime", "portal", "complaint"],
    },
    {
        "query": "What is Right to Information Act?",
        "keywords": ["rti", "information", "public", "pio", "appeal"],
    },
    {
        "query": "What is Article 21 right to life?",
        "keywords": ["article 21", "life", "liberty", "dignity", "constitution"],
    },
    {
        "query": "What is the punishment for theft under IPC?",
        "keywords": ["theft", "ipc", "section 379", "imprisonment", "three years"],
    },
    {
        "query": "How to file a complaint for domestic violence?",
        "keywords": ["domestic violence", "protection officer", "pwdva", "magistrate"],
    },
    {
        "query": "What are the rights of a woman under Indian law?",
        "keywords": ["woman", "rights", "posh", "maternity", "domestic violence"],
    },
    {
        "query": "What is bail and how to get bail?",
        "keywords": ["bail", "anticipatory bail", "session", "magistrate", "bnss"],
    },
    {
        "query": "What is cyber crime and how to report it?",
        "keywords": ["cybercrime", "1930", "it act", "report", "portal"],
    },
    {
        "query": "What is the procedure for divorce in India?",
        "keywords": ["divorce", "mutual consent", "family court", "alimony", "grounds"],
    },
    {
        "query": "What are consumer rights in India?",
        "keywords": ["consumer", "rights", "redressal", "district commission", "2019"],
    },
    {
        "query": "What is the Right to Education Act?",
        "keywords": ["education", "rte", "article 21a", "6 to 14", "free"],
    },
    {
        "query": "What is sedition law in India?",
        "keywords": ["sedition", "124a", "bns 152", "disaffection", "sovereignty"],
    },
    {
        "query": "How to register a property in India?",
        "keywords": ["registration", "stamp duty", "sub-registrar", "sale deed", "rera"],
    },
    {
        "query": "What is POCSO Act?",
        "keywords": ["pocso", "child", "sexual", "18 years", "special court"],
    },
]


def _hit(docs, keywords: list[str]) -> bool:
    """Return True if at least one retrieved doc contains ≥1 expected keyword."""
    combined = " ".join(d.page_content.lower() for d in docs)
    return any(kw.lower() in combined for kw in keywords)


def main():
    print("Loading embeddings model (this may take ~30s first time)...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Loading FAISS index...")
    db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    print("Loading BM25 index...")
    bm25_file = Path(BM25_PATH) / "bm25.pkl"
    bm25 = bm25_docs = None
    if bm25_file.exists():
        with open(bm25_file, "rb") as f:
            data = pickle.load(f)
        bm25 = data["bm25"]
        bm25_docs = data["docs"]
        print(f"BM25 loaded: {len(bm25_docs)} documents")
    else:
        print("BM25 index not found — dense-only mode")

    k = TOP_K
    print(f"\n{'='*60}")
    print(f"Running evaluation on {len(QUERIES)} queries | TOP_K={k}")
    print(f"{'='*60}\n")

    dense_hits = 0
    hybrid_hits = 0
    results = []

    for i, item in enumerate(QUERIES, 1):
        q = item["query"]
        kws = item["keywords"]

        # Dense retrieval
        dense_docs = db.similarity_search(q, k=k)
        dense_hit = _hit(dense_docs, kws)

        # Hybrid retrieval (RRF)
        if bm25 is not None:
            tokens = q.lower().split()
            scores = bm25.get_scores(tokens)
            top_idx = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:k]
            sparse_docs = [bm25_docs[j] for j in top_idx]

            doc_scores: dict[str, float] = {}
            doc_map: dict[str, Document] = {}
            alpha = 0.7
            for rank, doc in enumerate(dense_docs):
                key = doc.page_content[:100]
                doc_scores[key] = doc_scores.get(key, 0) + alpha * (1 / (rank + 1))
                doc_map[key] = doc
            for rank, doc in enumerate(sparse_docs):
                key = doc.page_content[:100]
                doc_scores[key] = doc_scores.get(key, 0) + (1 - alpha) * (1 / (rank + 1))
                doc_map[key] = doc
            merged = [doc_map[ky] for ky in sorted(doc_scores, key=doc_scores.__getitem__, reverse=True)[:k]]
        else:
            merged = dense_docs

        hybrid_hit = _hit(merged, kws)

        if dense_hit:
            dense_hits += 1
        if hybrid_hit:
            hybrid_hits += 1

        status = "✓ both" if (dense_hit and hybrid_hit) else ("✓ hybrid only" if hybrid_hit else ("✓ dense only" if dense_hit else "✗ both missed"))
        print(f"[{i:2}] {q[:55]:<55} Dense={'✓' if dense_hit else '✗'}  Hybrid={'✓' if hybrid_hit else '✗'}  {status}")

        results.append({
            "query": q,
            "dense_hit": dense_hit,
            "hybrid_hit": hybrid_hit,
        })

    total = len(QUERIES)
    dense_pct = 100 * dense_hits / total
    hybrid_pct = 100 * hybrid_hits / total
    improvement = hybrid_pct - dense_pct

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Dense-only  Hit Rate @{k}: {dense_hits}/{total}  = {dense_pct:.1f}%")
    print(f"Hybrid      Hit Rate @{k}: {hybrid_hits}/{total}  = {hybrid_pct:.1f}%")
    print(f"Improvement:                  {improvement:+.1f}% (hybrid over dense-only)")
    print(f"{'='*60}\n")

    # ── SQLite memory test ─────────────────────────────────────────────────────
    print("Testing SQLite session memory...")
    try:
        from memory import init_db, save_turn, get_history, clear_session
        init_db()
        sid = "eval_test_session"
        mid = save_turn(sid, "test question", "test answer")
        hist = get_history(sid)
        assert len(hist) == 2, f"Expected 2 history rows, got {len(hist)}"
        clear_session(sid)
        assert get_history(sid) == [], "Session not cleared"
        print("Session memory: ✅ OK")
    except Exception as e:
        print(f"Session memory: ✗ ERROR — {e}")

    # ── Save results ───────────────────────────────────────────────────────────
    out = Path(__file__).parent / "eval_results.json"
    with open(out, "w") as f:
        json.dump({
            "total": total,
            "top_k": k,
            "dense_hit_rate": round(dense_pct, 1),
            "hybrid_hit_rate": round(hybrid_pct, 1),
            "improvement": round(improvement, 1),
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to: {out}")

    print("\n── USE THESE NUMBERS IN YOUR RESUME ──────────────────────")
    print(f"Hybrid FAISS+BM25 retrieval improved recall by {improvement:.0f}%")
    print(f"over pure semantic search (Hit Rate: {hybrid_pct:.0f}% vs {dense_pct:.0f}%)")
    print(f"on 15 Indian legal queries @ Top-{k}")
    print("──────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
