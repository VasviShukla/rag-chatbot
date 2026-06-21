# RAG Chatbot — Chat With Your Own Documents

A production-style Retrieval-Augmented Generation (RAG) chatbot: upload your
own PDFs/Word docs/text files, and ask questions about them in a chat
interface with cited sources, streaming responses, and persistent
conversation history.

Built with **LangChain + FastAPI**, using **only free services** — no paid
API keys, no hosted vector DB subscription, no GPU required.

```
PDF / DOCX / TXT  →  chunk  →  embed (local)  →  Chroma (local)
                                                       │
User question  →  rewrite w/ history  →  retrieve  ───┘
                                                       │
                                          stuff into prompt
                                                       │
                                     Groq LLM (free tier, streamed)
                                                       │
                                  Answer + cited source chunks
```

## Why this project

This repo is intentionally built the way a small production RAG service
would be, not a notebook demo:

- **Conversational RAG**, not single-turn Q&A — a history-aware retriever
  rewrites follow-up questions ("what about its pricing?") into standalone
  queries before retrieval, using LangChain's LCEL chain composition
  (`create_history_aware_retriever` + `create_stuff_documents_chain` +
  `create_retrieval_chain`).
- **Source attribution** — every answer returns the exact chunks (filename,
  chunk index, page number) it was generated from, so answers are
  verifiable instead of opaque.
- **Streaming** — answers stream token-by-token over Server-Sent Events,
  consumed by a Streamlit UI with a live-typing effect.
- **Persistence with zero extra infra** — SQLite for document metadata and
  chat history, Chroma (embedded, file-backed) for vectors. No external
  database to provision, no cost.
- **Fully free inference stack** — embeddings run locally on CPU
  (sentence-transformers), and the LLM call goes to Groq's free-tier API
  (fast Llama 3.3 inference, no credit card required).
- **Tested** — pytest suite covering ingestion/chunking and the full HTTP
  API surface, with the LLM and embedding model faked out so CI runs in
  seconds with no network access or API key.
- **Containerized** — `docker-compose up` runs backend + frontend together.

## Architecture

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | async, typed, auto-generated OpenAPI docs at `/docs` |
| Orchestration | LangChain (LCEL) | history-aware retrieval chain, swappable LLM/retriever |
| LLM | Groq (`llama-3.3-70b-versatile`) | free tier, very low latency, OpenAI-compatible |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | local, free, no API key, runs on CPU |
| Vector store | Chroma (persisted to disk) | embedded, free, no hosted service |
| Metadata / history | SQLite | zero-infra, durable across restarts |
| Frontend | Streamlit | fast to build, calls the API over plain HTTP |

```
rag-chatbot/
├── app/
│   ├── main.py          FastAPI routes
│   ├── config.py        typed settings (env-driven)
│   ├── db.py             SQLite: document metadata + chat history
│   ├── ingestion.py     loaders + chunking (PDF/TXT/MD/DOCX)
│   ├── vectorstore.py   Chroma + embeddings access layer
│   ├── rag_chain.py     LangChain RAG pipeline (sync + streaming)
│   ├── models.py        Pydantic request/response schemas
│   └── logger.py
├── frontend/
│   └── streamlit_app.py  chat UI (HTTP client only, no LLM logic)
├── tests/
│   ├── conftest.py       fakes for embeddings/LLM, isolated test env
│   ├── test_ingestion.py
│   └── test_api.py
├── sample_docs/           a doc to try the demo with immediately
├── Dockerfile / Dockerfile.frontend / docker-compose.yml
├── Makefile
└── requirements*.txt
```


## API reference

Interactive docs are auto-generated at `/docs` (Swagger) and `/redoc`. Summary:

| Method | Path | Description |
|---|---|---|
| GET | `/health` | service status + index stats |
| POST | `/api/documents/upload` | multipart file upload (pdf/txt/md/docx) |
| POST | `/api/documents/ingest` | chunk + embed uploaded doc(s) into Chroma |
| GET | `/api/documents` | list documents and ingestion status |
| DELETE | `/api/documents/{doc_id}` | remove a document and its vectors |
| POST | `/api/chat` | ask a question, get the full answer + sources |
| POST | `/api/chat/stream` | same, but Server-Sent Events token stream |
| GET | `/api/chat/history/{session_id}` | fetch a conversation's history |
| DELETE | `/api/chat/history/{session_id}` | clear a conversation |

Example:

```bash
curl -F "file=@sample_docs/company_handbook.txt" \
  http://localhost:8000/api/documents/upload

curl -X POST http://localhost:8000/api/documents/ingest -H "Content-Type: application/json" -d '{}'

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "How much PTO do employees get?"}'
```




## Possible extensions

- Swap Chroma for a managed vector DB (e.g. Pinecone/Qdrant Cloud free tier)
  to demonstrate horizontal scaling.
- Add re-ranking (e.g. a cross-encoder) on top of the initial vector search.
- Add RAGAS-based automated evaluation (faithfulness, answer relevancy) as
  a CI gate.
- Add auth (API keys or OAuth) and per-user document isolation.
- Swap SQLite chat history for Redis if running multiple backend replicas.

