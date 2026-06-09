# NyayaSathi — AI Legal Assistant for Indian Law

> An RAG-powered legal Q&A system that answers questions on Indian law in plain language, with inline citations to exact sections and articles.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What It Does

NyayaSathi lets anyone ask questions about Indian law in plain English and get accurate, cited answers — no legal background needed.

**Example queries it handles:**
- *"What are my rights if police arrests me without a warrant?"*
- *"How do I file an FIR online?"*
- *"What is bail and how do I get it?"*
- *"What is the punishment for theft under IPC?"*
- *"How to file a complaint for domestic violence?"*

Every answer cites the exact law — e.g. `(Article 22, Constitution)` or `(IPC §379)` — so users can verify.

---

## Evaluation Results

| Metric | Score |
|---|---|
| Dense (FAISS) Hit Rate @Top-5 | **100%** (15/15 queries) |
| Hybrid (FAISS + BM25) Hit Rate @Top-5 | **100%** (15/15 queries) |
| Corpus size | 116 chunks across 17 legal domains |
| Session memory | SQLite-backed multi-turn |

Run the eval yourself:
```bash
python backend/evaluate_retrieval.py
```

---

## Architecture

```
User Query
    │
    ▼
HyDE Query Expansion  ←─── LLM generates hypothetical legal passage
    │
    ▼
Hybrid Retrieval
 ├── FAISS (dense semantic search)
 └── BM25  (sparse keyword search)
    │
    ▼
Reciprocal Rank Fusion (RRF)
    │
    ▼
Top-K Chunks → LLM (Groq / OpenRouter / Ollama)
    │
    ▼
Answer with inline citations + confidence score
```

**Key components:**
- **HyDE** — before retrieval, the LLM writes a hypothetical legal document to improve semantic match on jurisdiction-specific queries
- **Hybrid FAISS + BM25** — dense vectors catch semantic similarity; BM25 catches exact legal terms (section numbers, act names)
- **Reciprocal Rank Fusion** — merges both result lists without needing score calibration
- **SQLite memory** — each session stores conversation history; last N turns injected into the prompt

---

## Legal Corpus — 17 Domains Covered

| File | Topics |
|---|---|
| `constitution_articles.txt` | Articles 14, 19, 21, 22, 32, 39A, 226 |
| `ipc_sections.txt` | IPC 302, 354, 376, 379, 420, 498A and more |
| `bns_sections.txt` | BNS 2023 + BNSS 2023 (replaces IPC/CrPC from July 2024) |
| `bail_arrest_rights.txt` | D.K. Basu guidelines, types of bail, 24-hour rule |
| `fir_police_complaints.txt` | Online FIR, Zero FIR, e-FIR portals, state-wise links |
| `women_rights.txt` | PWDVA, POSH Act, Maternity Act, dowry law |
| `pocso_act.txt` | Child sexual offences, mandatory reporting, special courts |
| `cybercrime.txt` | IT Act sections, 1930 helpline, reporting steps |
| `consumer_rights.txt` | Consumer Protection Act 2019, e-Daakhil portal |
| `divorce_family_law.txt` | Hindu/Muslim/Christian divorce, mutual consent procedure |
| `rti_act.txt` | RTI filing, online portal, appeals, penalties |
| `right_to_education.txt` | RTE Act, 25% quota, Article 21A |
| `sedition_law.txt` | IPC 124A vs BNS Section 152 comparison |
| `property_registration.txt` | Stamp duty, Sub-Registrar steps, RERA |
| `legal_aid.txt` | NALSA, DLSA, free legal aid eligibility |
| `tele_law_scheme.txt` | DOJ Tele-Law scheme for rural legal access |
| `cybercrime.txt` | Cybercrime reporting, IT Act offences |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| LLM | Groq (default) / OpenRouter / Ollama |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense Index | FAISS |
| Sparse Index | BM25 (rank-bm25) |
| Memory | SQLite |
| Orchestration | LangChain |

---

## Project Structure

```
nyayasathi/
├── backend/
│   ├── main.py              # FastAPI app — all routes
│   ├── rag.py               # Hybrid retrieval + HyDE + LLM pipeline
│   ├── ingestion.py         # Build FAISS + BM25 indexes from corpus
│   ├── memory.py            # SQLite session memory
│   ├── config.py            # Central config (LLM, paths, params)
│   ├── models.py            # Pydantic request/response models
│   ├── evaluate_retrieval.py# Retrieval eval — Hit Rate @Top-5
│   └── requirements.txt
├── data/
│   ├── corpus/              # 17 legal domain .txt files
│   ├── faiss_index/         # Built by ingestion.py
│   └── bm25_index/          # Built by ingestion.py
└── frontend/
    ├── index.html
    ├── scripts/app.js
    └── styles/style.css
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/swastik2475/Nyaya_Sathi.git
cd Nyaya_Sathi
pip install -r backend/requirements.txt
```

### 2. Configure environment

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:
```env
LLM_PROVIDER=groq          # groq | openrouter | ollama
GROQ_API_KEY=your_key_here # free at console.groq.com
```

### 3. Build the indexes

```bash
python backend/ingestion.py
```

This loads the corpus, scrapes 7 legal websites, and builds FAISS + BM25 indexes. Takes ~2 minutes on first run.

### 4. Start the API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Main RAG query — returns answer + sources + confidence |
| `GET` | `/history/{session_id}` | Fetch chat history for a session |
| `DELETE` | `/history/{session_id}` | Clear session (new chat) |
| `POST` | `/feedback` | Store 👍/👎 rating for a response |
| `GET` | `/health` | Liveness check + model info |

**Example request:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is bail?", "session_id": "abc123"}'
```

**Example response:**
```json
{
  "answer": "Bail is the temporary release of an accused person... (BNSS §480)",
  "sources": ["bail_arrest_rights.txt"],
  "confidence": 0.87,
  "message_id": "uuid-here"
}
```

---

## LLM Provider Options

| Provider | Model | Cost | Setup |
|---|---|---|---|
| **Groq** (default) | `llama-3.1-70b-versatile` | Free (14,400 req/day) | [console.groq.com](https://console.groq.com) |
| OpenRouter | `llama-3.1-8b-instruct:free` | Free tier available | [openrouter.ai](https://openrouter.ai) |
| Ollama | `gemma2:2b` | Fully local, no API key | [ollama.ai](https://ollama.ai) |

---

## Achievements

- 🏆 **Top 5 — Smart India Hackathon (SIH) 2024** Internal Round for AI-based Legal Assistant
- 📊 **100% Hit Rate @Top-5** on 15 Indian legal benchmark queries

---

## Emergency Legal Helplines

| Service | Number |
|---|---|
| Police | 112 |
| NALSA Legal Aid | 1800-180-1516 |
| Cybercrime | 1930 |
| Women Helpline | 181 |
| Childline | 1098 |

---

## Disclaimer

NyayaSathi provides general legal information only. It is not a substitute for advice from a qualified advocate. For serious legal matters, always consult a licensed lawyer or contact your nearest District Legal Services Authority (DLSA).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
