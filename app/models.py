"""Pydantic schemas for API requests and responses."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    status: str  # "uploaded" | "ingesting" | "ingested" | "failed"
    chunk_count: int = 0
    uploaded_at: datetime
    ingested_at: Optional[datetime] = None
    error: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class IngestRequest(BaseModel):
    doc_ids: Optional[list[str]] = Field(
        default=None,
        description="Specific document IDs to ingest. Omit to ingest all documents "
        "currently in 'uploaded' or 'failed' status.",
    )


class IngestResult(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int = 0
    error: Optional[str] = None


class IngestResponse(BaseModel):
    results: list[IngestResult]


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated conversation/session identifier")
    message: str = Field(..., min_length=1, max_length=4000)


class SourceChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    page: Optional[int] = None
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk]


class HistoryMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class HealthResponse(BaseModel):
    status: str
    documents_ingested: int
    total_chunks: int
    llm_model: str
    embedding_model: str
