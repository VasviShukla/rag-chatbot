"""
Vector store access layer (Chroma, persisted to local disk — free, no
hosted vector DB required).

Embeddings are computed locally with a small sentence-transformers model
so there is no API cost or network dependency for the embedding step;
only the LLM call itself goes out to Groq's free-tier API.
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def get_vectorstore() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.CHROMA_PERSIST_DIR),
    )


def add_chunks(chunks) -> int:
    if not chunks:
        return 0
    ids = [f"{c.metadata['doc_id']}::{c.metadata['chunk_index']}" for c in chunks]
    get_vectorstore().add_documents(documents=chunks, ids=ids)
    return len(chunks)


def delete_by_doc_id(doc_id: str) -> None:
    store = get_vectorstore()
    store.delete(where={"doc_id": doc_id})


def get_retriever(k: int | None = None):
    settings = get_settings()
    return get_vectorstore().as_retriever(
        search_kwargs={"k": k or settings.RETRIEVER_TOP_K}
    )


def collection_chunk_count() -> int:
    try:
        return get_vectorstore()._collection.count()  # noqa: SLF001 (no public count API)
    except Exception:  # pragma: no cover - defensive
        return 0
