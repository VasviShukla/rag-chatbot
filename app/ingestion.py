"""
Document loading and chunking.

Supports PDF, TXT/Markdown, and DOCX out of the box. Each loader returns
LangChain `Document` objects; we then split them with a recursive
character splitter tuned for prose, and stamp every chunk with metadata
(doc_id, filename, chunk_index, page) so answers can be traced back to an
exact chunk in an exact source file later (citation/source-attribution).
"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)


class UnsupportedFileType(ValueError):
    pass


def load_document(path: Path) -> list[Document]:
    """Dispatch to the right LangChain loader based on file extension."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path)).load()

    if suffix in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(path), encoding="utf-8").load()

    if suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(str(path)).load()

    raise UnsupportedFileType(f"Unsupported file type: {suffix}")


def split_documents(docs: list[Document], doc_id: str, filename: str) -> list[Document]:
    """Split raw documents into retrieval-sized chunks with citation metadata."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        # PyPDFLoader already sets metadata["page"]; normalize so it's always present.
        page = chunk.metadata.get("page")
        chunk.metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "page": page,
        }

    logger.info("Split %s into %d chunks (doc_id=%s)", filename, len(chunks), doc_id)
    return chunks


def load_and_split(path: Path, doc_id: str, filename: str) -> list[Document]:
    raw_docs = load_document(path)
    if not raw_docs or not any(d.page_content.strip() for d in raw_docs):
        raise ValueError("Document contains no extractable text")
    return split_documents(raw_docs, doc_id=doc_id, filename=filename)
