# 🔍 Grounded AI Search Assistant

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Railway-6c63ff?style=for-the-badge&logo=railway)](https://grounded-ai-assistant-production.up.railway.app/ui)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Corpus](https://img.shields.io/badge/Corpus-304%20chunks-00c9a7?style=for-the-badge)](#corpus)
[![Deployed](https://img.shields.io/badge/Deployed-Production-success?style=for-the-badge)](#)

> **Production-style RAG system** that grounds every LLM answer in real arXiv research papers and UK Government AI policy documents — with real-time streaming, clickable source citations, and retrieval evaluation.

---

## 🚀 Live Demo

**Try it now →** [grounded-ai-assistant-production.up.railway.app/ui](https://grounded-ai-assistant-production.up.railway.app/ui)

> ⏱ May take ~20s to wake on first request (Railway free tier sleep mode)

![Demo Screenshot](https://raw.githubusercontent.com/DiyaMatthew/grounded-ai-assistant/main/demo.png)
<img width="1640" height="934" alt="Screenshot 2026-04-23 at 12 06 38" src="https://github.com/user-attachments/assets/e67e6fcb-e45a-4fcd-bbcd-372d7ebe85a4" />
<img width="1637" height="932" alt="Screenshot 2026-04-23 at 13 25 16" src="https://github.com/user-attachments/assets/2792bd5a-3941-4655-a05a-8f0b2109929c" />





---

## 🧠 What This Project Demonstrates

This is not a chatbot. It is a **production-pattern Retrieval-Augmented Generation (RAG) pipeline** that demonstrates the core skills demanded by AI engineering roles:

| Skill | Implementation |
|---|---|
| **RAG pipeline** | Corpus ingestion → chunking → embedding → vector search → LLM generation |
| **Semantic search** | `all-MiniLM-L6-v2` bi-encoder with cosine similarity ranking |
| **Real-time streaming** | Server-Sent Events (SSE) token-by-token streaming via FastAPI |
| **Source grounding** | Every answer cites retrieved documents with relevance scores |
| **Production deployment** | Dockerised, deployed on Railway with environment variable secrets |
| **Multi-source retrieval** | Hybrid corpus: academic papers (arXiv) + policy documents (UK Gov) |
| **Session memory** | Per-session conversation history for follow-up questions |

---
## 💡 Why This Project Exists

AI is moving extremely fast — new regulations, new research, and new 
terminology every few months. Businesses and organisations, especially 
in the UK, are struggling to keep up. Understanding things like the 
UK's National AI Strategy, the pro-innovation approach to AI regulation, 
or what a large language model actually is means digging through 
government white papers and academic research papers — which takes time 
and expertise most teams don't have.

Grounded AI Assistant solves that by letting anyone ask a plain English 
question and get an answer sourced directly from real UK Government 
documents and arXiv research papers — with citations so you can verify 
every claim. No hallucinations. No guesswork. Just grounded answers.

---
## 🏗 Architecture

```
User Query
    │
    ▼
Frontend (JS + SSE)
    │
    ▼
FastAPI Backend  ──────────────────────────────────────┐
    │                                                   │
    ▼                                                   ▼
VectorStore.search()                          Session Memory
    │                                         (per session_id)
    ├── Embed query (MiniLM)
    ├── Cosine similarity over 304 chunks
    └── Return top-3 with metadata
    │
    ▼
format_context() → inject into LLM prompt
    │
    ▼
Groq API (llama-3.1-8b-instant)
    │  streaming
    ▼
SSE stream → Frontend renders tokens + citations
```

---

## 📚 Corpus

| Source | Content | Chunks |
|---|---|---|
| **arXiv** | Research papers on LLMs, RAG, transformers, RLHF, AI safety, diffusion models | ~290 |
| **UK Gov** | National AI Strategy, ICO AI guidance, NHS AI Lab, AI regulation white paper | ~14 |
| **Total** | 300 documents → **304 retrieval chunks** | **304** |

Built with `python data/build_corpus.py`. Corpus is pre-built and committed — zero cold-start data dependency.

---

## ⚙️ Tech Stack

**Backend**
- `FastAPI` — async Python web framework
- `sentence-transformers` — `all-MiniLM-L6-v2` for dense embeddings
- `numpy` — cosine similarity search (in-memory vector store)
- `Groq` — LLM inference (`llama-3.1-8b-instant`)
- `Server-Sent Events` — real-time token streaming

**Data Ingestion**
- `arxiv` — programmatic paper fetching across 10 AI topic queries
- `requests` + `BeautifulSoup` — UK Gov policy page scraping
- Custom chunking pipeline with overlap for retrieval quality

**Frontend**
- Vanilla JS with `EventSource` API for SSE streaming
- Citation cards with relevance score bars
- Session-aware follow-up suggestions

**Infrastructure**
- Railway (production deployment)
- Environment variable secrets management
- CPU-only PyTorch for sub-4GB image size

---

## 📊 Retrieval Quality

Sample eval results on 3 test queries:

| Query | Top Source | Relevance Score |
|---|---|---|
| "What is retrieval-augmented generation?" | Ragas: Automated Evaluation of RAG | 57% |
| "How does the UK regulate AI?" | UK National AI Strategy | 72% |
| "Explain transformer attention" | Engineering the RAG Stack | 57% |

Run evals locally: `python eval/test_retrieval.py`

---

## 🔧 Run Locally

```bash
# Clone
git clone https://github.com/DiyaMatthew/grounded-ai-assistant.git
cd grounded-ai-assistant

# Set up environment (requires Python 3.11)
conda create -n rag-env python=3.11 -y
conda activate rag-env
pip install -r requirements.txt

# Add API key
echo "GROQ_API_KEY=your_key_here" > .env

# Start server
uvicorn backend.app:app --reload

# Open UI
open http://localhost:8000/ui
```

---

## 📁 Project Structure

```
grounded-ai-assistant/
├── backend/
│   ├── app.py          # FastAPI routes + SSE streaming endpoint
│   ├── rag.py          # VectorStore: embed, index, search, deduplicate
│   └── llm.py          # Groq streaming client + session memory
├── data/
│   ├── ingest_arxiv.py     # arXiv paper fetcher (10 topic queries)
│   ├── ingest_gov_docs.py  # UK Gov policy page scraper
│   ├── build_corpus.py     # Merge, chunk, deduplicate → corpus.json
│   └── corpus.json         # Pre-built corpus (304 chunks)
├── frontend/
│   └── index.html      # SSE client, citation renderer, session UI
├── eval/
│   └── test_retrieval.py   # Retrieval precision test suite
├── railway.toml        # Production deployment config
└── requirements.txt
```

---

## 🔮 Planned Improvements

- [ ] FAISS or Pinecone for scalable vector storage (>100k chunks)
- [ ] BM25 hybrid search for sparse + dense retrieval
- [ ] Redis session memory for multi-instance deployments
- [ ] LLM-as-judge eval scoring (faithfulness + relevance)
- [ ] Authentication + rate limiting for public API

---

## 👩‍💻 Author

**Diya Mathew**   
[GitHub](https://github.com/DiyaMatthew) · [LinkedIn](https://www.linkedin.com/in/diya-mathew/)

---

*Built as a portfolio project demonstrating production RAG engineering skills.*
