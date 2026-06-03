# ⚖️ NyayaSathi — Indian Legal AI Assistant

> **"Sabko Nyay, Har Ghar Nyaya"**  
> Ask any question about Indian law — FIR, rights, schemes, IPC sections



---

## 🚀 What is NyayaSathi?

NyayaSathi is a **RAG-powered (Retrieval-Augmented Generation) legal AI assistant** built for Indian citizens. It answers questions about Indian law — criminal, civil, constitutional, RTI, and cyber — in simple English or Hindi, with inline citations from actual legal documents.

Built as a **Top 5 project at SIH 2024 (Smart India Hackathon)** internal round.

---

## ✨ Features

- 🔍 **Hybrid Retrieval** — FAISS (dense) + BM25 (sparse) with reciprocal rank fusion
- 🧠 **HyDE Query Expansion** — generates hypothetical legal documents to improve semantic recall
- 💬 **Multi-turn Memory** — session-based conversation history with SQLite
- 📚 **Source Citations** — every answer cites the relevant law/section inline
- 🔌 **Pluggable LLM** — Groq (default) → OpenRouter → Ollama fallback
- 👍👎 **Feedback System** — helpfulness ratings stored per response
- 🌐 **Clean Frontend** — vanilla JS single-page app, no framework needed

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq API (`llama-3.3-70b-versatile`) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (dense retrieval) |
| **Sparse Retrieval** | BM25 (rank-bm25) |
| **RAG Framework** | LangChain |
| **Backend** | FastAPI + Uvicorn |
| **Memory** | SQLite |
| **Frontend** | HTML + CSS + Vanilla JS |

---

## 📁 Project Structure

```
nyayasathi/
├── backend/
│   ├── main.py          # FastAPI app — routes & startup
│   ├── rag.py           # RAG pipeline — retrieval + LLM
│   ├── config.py        # All env vars & model config
│   ├── memory.py        # SQLite chat history
│   ├── models.py        # Pydantic request/response models
│   ├── ingestion.py     # PDF ingestion → FAISS + BM25 index
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── scripts/app.js
│   └── styles/style.css
├── data/
│   ├── corpus/          # Legal PDFs (not tracked in git)
│   ├── faiss_index/     # Built by ingestion.py
│   └── bm25_index/      # Built by ingestion.py
└── docs/
```

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/swastik2475/Nyaya_Sathi.git
cd Nyaya_Sathi
```

### 2. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Build the index
```bash
# Add your legal PDFs to data/corpus/
python ingestion.py
```

### 5. Start the backend
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open the frontend
Open `frontend/index.html` in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness + readiness check |
| `POST` | `/query` | Main RAG query |
| `GET` | `/history/{session}` | Fetch chat history |
| `DELETE` | `/history/{session}` | Clear session |
| `POST` | `/feedback` | Store 👍/👎 rating |

---

## 🧩 RAG Pipeline

```
User Query
    │
    ▼
HyDE Expansion (LLM generates hypothetical legal text)
    │
    ├──► FAISS Dense Retrieval
    │
    └──► BM25 Sparse Retrieval
              │
              ▼
         Reciprocal Rank Fusion
              │
              ▼
         Top-K Chunks + Context
              │
              ▼
         LLM Answer Generation
              │
              ▼
         Answer + Citations + Confidence Score
```

---

## 🔑 LLM Providers

| Provider | Model | Notes |
|----------|-------|-------|
| **Groq** (default) | `llama-3.3-70b-versatile` | Free, 14,400 req/day |
| OpenRouter | configurable | Free models available |
| Ollama | `gemma2:2b` | Fully local, no API key |

Switch provider in `.env`: `LLM_PROVIDER=groq|openrouter|ollama`

---

## 👨‍💻 Author

**Swastik** — B.Tech AI, BIT Mesra (Jaipur Campus) '27  
[GitHub](https://github.com/swastik2475)

---

## ⚠️ Disclaimer

NyayaSathi provides general legal information only. It is **not a substitute for professional legal advice**. For serious legal matters, consult a qualified advocate. Dial **112** (Police) or **1516** (Legal Aid) for emergencies.
