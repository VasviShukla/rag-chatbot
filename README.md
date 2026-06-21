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

## Quickstart (local, no Docker)

**1. Get a free Groq API key** — sign up at
[console.groq.com/keys](https://console.groq.com/keys) (no credit card
required) and copy your key.

**2. Set up the environment**

```bash
git clone <this-repo>
cd rag-chatbot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
cp .env.example .env
# edit .env and paste your GROQ_API_KEY
make install
```

**3. Run the backend**

```bash
make dev
# API docs: http://localhost:8000/docs
```

**4. Run the chat UI** (in a second terminal)

```bash
make install-frontend
make run-frontend
# UI: http://localhost:8501
```

**5. Try it** — in the UI sidebar, upload `sample_docs/company_handbook.txt`,
click "Ingest", then ask: *"How many days of PTO do new hires get?"* or
*"What's the home-office equipment stipend?"*

## Quickstart (Docker)

```bash
cp .env.example .env   # add your GROQ_API_KEY
docker compose up --build
# API:  http://localhost:8000/docs
# Chat UI: http://localhost:8501
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

## Deployment (free hosting)

**Backend → [Render](https://render.com) free web service.**
The repo includes a `render.yaml` Blueprint, so the easiest path is:

1. Push this repo to GitHub (see below if you haven't yet).
2. In Render: **New → Blueprint**, connect the repo, and it'll detect
   `render.yaml` automatically.
3. When prompted, set the `GROQ_API_KEY` secret (it's intentionally not
   committed to the repo).
4. Deploy. Your API will be live at `https://<service-name>.onrender.com`
   — confirm with `https://<service-name>.onrender.com/health`.

   *No Blueprint?* You can also do it manually: **New → Web Service** →
   connect the repo → Environment: **Docker** → plan: **Free** → add the
   same env vars from `render.yaml` → set health check path to `/health`.

**Frontend → [Streamlit Community Cloud](https://share.streamlit.io).**

1. **New app** → select this repo/branch → **Main file path:**
   `frontend/streamlit_app.py` (Streamlit Cloud will pick up
   `frontend/requirements.txt` automatically since it sits next to the
   entrypoint).
2. In **Advanced settings → Secrets**, add:
   ```toml
   BACKEND_URL = "https://<your-render-service>.onrender.com"
   ```
3. Deploy. Your chat UI will be live at `https://<your-app>.streamlit.app`.
4. (Optional, recommended) Back in Render, set `CORS_ORIGINS` to your
   Streamlit app's exact URL instead of `*`.

### Free-tier realities worth knowing about

- **Render's free web services have no persistent disk.** The SQLite
  database and Chroma vector store live on local disk, so uploaded
  documents and chat history are wiped whenever the instance restarts or
  redeploys. For a portfolio demo this is an acceptable, well-understood
  trade-off — just re-upload `sample_docs/company_handbook.txt` after a
  cold start. (A natural next step: move document/chat metadata to
  Render's free Postgres and the vectors to a free-tier hosted vector DB.)
- **Free instances spin down after 15 minutes of inactivity** and take
  ~30-60s to wake back up on the next request — the first request after
  idle time will be slow, that's expected.
- **512MB RAM is tight** for `sentence-transformers` + `torch`, even
  CPU-only. The Dockerfile sets `OMP_NUM_THREADS=1` to reduce memory/CPU
  contention. If you see out-of-memory restarts in the Render logs, the
  cheapest fixes are (a) upgrade to Render's $7/mo Starter plan (1 GB+
  RAM), or (b) swap local embeddings for a free hosted embeddings API
  (e.g. the HuggingFace Inference API) so `torch` never loads in the
  deployed process.
- **Streamlit Community Cloud apps sleep after 12 hours with no
  traffic** and need a click to wake up — fine for a resume link people
  visit occasionally.

## Pushing this to GitHub

```bash
cd rag-chatbot
git init                       # skip if already a git repo
git add .
git commit -m "Initial commit: RAG chatbot (LangChain + FastAPI)"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

`.env` is already in `.gitignore` — double-check your real Groq key never
gets committed before you push.

## Running tests

```bash
make install   # installs requirements-dev.txt
make test
```

The test suite never calls Groq or downloads an embedding model — both are
replaced with lightweight fakes in `tests/conftest.py` — so it runs fully
offline in a few seconds.

## Design notes & trade-offs

- **Why Groq instead of OpenAI?** Groq's free tier needs no credit card and
  has very low latency, which makes the streaming demo feel snappy. The
  `get_llm()` function in `app/rag_chain.py` is the only place tied to
  Groq — swapping to another `langchain` chat model is a one-function change.
- **Why local embeddings instead of a hosted embeddings API?** Removes a
  second network dependency and API cost entirely; `all-MiniLM-L6-v2` is
  small enough to run on a laptop CPU in real time.
- **Why SQLite, not Postgres/Redis?** This is a single-process service. A
  zero-infra embedded database keeps the project free and easy to run
  anywhere, while still using real SQL and transactions rather than
  in-memory dicts that vanish on restart.
- **Known limitation:** chat history and the vector index are not
  partitioned per-user — this is a single-tenant demo. Multi-tenant
  isolation (per-user collections, auth) is a natural next step.

## Possible extensions

- Swap Chroma for a managed vector DB (e.g. Pinecone/Qdrant Cloud free tier)
  to demonstrate horizontal scaling.
- Add re-ranking (e.g. a cross-encoder) on top of the initial vector search.
- Add RAGAS-based automated evaluation (faithfulness, answer relevancy) as
  a CI gate.
- Add auth (API keys or OAuth) and per-user document isolation.
- Swap SQLite chat history for Redis if running multiple backend replicas.

## Resume bullet points (for reference)

- Built a full-stack Retrieval-Augmented Generation chatbot (LangChain,
  FastAPI, Chroma) supporting multi-format document ingestion (PDF/DOCX/TXT),
  history-aware conversational retrieval, and token-level streaming over SSE.
- Designed a citation system that traces every generated answer back to the
  exact source chunk and page, improving answer verifiability.
- Achieved a zero-cost, zero-external-infra inference stack by combining
  local sentence-transformer embeddings, an embedded Chroma vector store,
  SQLite-backed persistence, and Groq's free-tier LLM API.
- Wrote a pytest suite covering ingestion and the full API surface using
  dependency-injected fakes for the LLM/embedding model, enabling fast,
  deterministic CI without network access.
- Containerized the service (FastAPI backend + Streamlit frontend) with
  Docker Compose for one-command local or cloud deployment.
