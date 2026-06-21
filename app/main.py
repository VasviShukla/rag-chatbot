"""
FastAPI application entrypoint.

Routes:
    GET    /health                          service + index health
    POST   /api/documents/upload            upload a file (pdf/txt/md/docx)
    POST   /api/documents/ingest            chunk + embed uploaded doc(s)
    GET    /api/documents                   list documents and their status
    DELETE /api/documents/{doc_id}          remove a document + its vectors
    POST   /api/chat                        ask a question (full response)
    POST   /api/chat/stream                 ask a question (SSE token stream)
    GET    /api/chat/history/{session_id}   fetch conversation history
    DELETE /api/chat/history/{session_id}   clear conversation history
"""
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import db
from app.config import get_settings
from app.ingestion import UnsupportedFileType, load_and_split
from app.logger import configure_logging, get_logger
from app.models import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    HistoryMessage,
    HistoryResponse,
    IngestRequest,
    IngestResponse,
    IngestResult,
    SourceChunk,
)
from app.rag_chain import get_chat_response, stream_chat_response
from app.vectorstore import add_chunks, collection_chunk_count, delete_by_doc_id

configure_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    logger.info("Database initialized at %s", settings.SQLITE_DB_PATH)
    yield


app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _row_to_document_info(row) -> DocumentInfo:
    return DocumentInfo(
        doc_id=row["doc_id"],
        filename=row["filename"],
        status=row["status"],
        chunk_count=row["chunk_count"],
        uploaded_at=row["uploaded_at"],
        ingested_at=row["ingested_at"],
        error=row["error"],
    )


# --------------------------------------------------------------------- health

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    docs_ingested, total_chunks = db.aggregate_stats()
    return HealthResponse(
        status="ok",
        documents_ingested=docs_ingested,
        total_chunks=total_chunks,
        llm_model=settings.GROQ_MODEL,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
    )


# ---------------------------------------------------------------- documents

@app.post("/api/documents/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    doc_id = uuid.uuid4().hex[:12]
    dest_path = settings.UPLOAD_DIR / f"{doc_id}{suffix}"

    size = 0
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    with dest_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"File exceeds the {settings.MAX_FILE_SIZE_MB}MB limit",
                )
            out.write(chunk)

    db.create_document(doc_id=doc_id, filename=file.filename, path=str(dest_path))
    logger.info("Uploaded document %s (%s, %d bytes)", doc_id, file.filename, size)
    return _row_to_document_info(db.get_document(doc_id))


@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    rows = db.list_documents()
    return DocumentListResponse(documents=[_row_to_document_info(r) for r in rows])


@app.post("/api/documents/ingest", response_model=IngestResponse)
def ingest_documents(payload: IngestRequest | None = None) -> IngestResponse:
    if payload and payload.doc_ids:
        rows = [db.get_document(doc_id) for doc_id in payload.doc_ids]
        rows = [r for r in rows if r is not None]
    else:
        rows = db.list_documents(statuses=["uploaded", "failed"])

    if not rows:
        return IngestResponse(results=[])

    results: list[IngestResult] = []
    for row in rows:
        doc_id, filename, path = row["doc_id"], row["filename"], row["path"]
        try:
            chunks = load_and_split(Path(path), doc_id=doc_id, filename=filename)
            n_added = add_chunks(chunks)
            db.set_document_status(doc_id, "ingested", chunk_count=n_added)
            results.append(
                IngestResult(doc_id=doc_id, filename=filename, status="ingested", chunk_count=n_added)
            )
            logger.info("Ingested %s: %d chunks", filename, n_added)
        except UnsupportedFileType as e:
            db.set_document_status(doc_id, "failed", error=str(e))
            results.append(IngestResult(doc_id=doc_id, filename=filename, status="failed", error=str(e)))
        except Exception as e:  # noqa: BLE001 - surface ingestion errors per-doc, don't crash the batch
            logger.exception("Failed to ingest %s", filename)
            db.set_document_status(doc_id, "failed", error=str(e))
            results.append(IngestResult(doc_id=doc_id, filename=filename, status="failed", error=str(e)))

    return IngestResponse(results=results)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    row = db.get_document(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")

    delete_by_doc_id(doc_id)
    Path(row["path"]).unlink(missing_ok=True)
    db.delete_document(doc_id)
    logger.info("Deleted document %s", doc_id)
    return {"deleted": doc_id}


# --------------------------------------------------------------------- chat

@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if collection_chunk_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been ingested yet. Upload and ingest a document first.",
        )
    try:
        result = get_chat_response(payload.session_id, payload.message)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return ChatResponse(
        session_id=payload.session_id,
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
    )


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest):
    if collection_chunk_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been ingested yet. Upload and ingest a document first.",
        )

    def event_generator():
        try:
            for event in stream_chat_response(payload.session_id, payload.message):
                yield f"data: {json.dumps(event)}\n\n"
        except RuntimeError as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/chat/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str) -> HistoryResponse:
    rows = db.get_chat_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[
            HistoryMessage(role=r["role"], content=r["content"], created_at=r["created_at"])
            for r in rows
        ],
    )


@app.delete("/api/chat/history/{session_id}")
def clear_history(session_id: str):
    db.clear_chat_history(session_id)
    return {"cleared": session_id}
