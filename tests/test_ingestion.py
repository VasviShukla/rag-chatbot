from pathlib import Path

from app.ingestion import UnsupportedFileType, load_and_split, load_document


def test_load_document_txt(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("Hello world. This is a test document about RAG chatbots.")

    docs = load_document(f)

    assert len(docs) == 1
    assert "RAG chatbots" in docs[0].page_content


def test_load_document_rejects_unsupported_extension(tmp_path: Path):
    f = tmp_path / "note.xyz"
    f.write_text("irrelevant")

    try:
        load_document(f)
        assert False, "expected UnsupportedFileType"
    except UnsupportedFileType:
        pass


def test_split_documents_attaches_citation_metadata(tmp_path: Path):
    f = tmp_path / "long.txt"
    # Long enough to require multiple chunks at the default 1000-char chunk size.
    f.write_text(("This is sentence number %d about RAG systems. " * 1) % 0 + "\n\n".join(
        f"Paragraph {i}: " + ("Lorem ipsum dolor sit amet. " * 20) for i in range(10)
    ))

    chunks = load_and_split(f, doc_id="doc-123", filename="long.txt")

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["doc_id"] == "doc-123"
        assert chunk.metadata["filename"] == "long.txt"
        assert chunk.metadata["chunk_index"] == i


def test_load_and_split_rejects_empty_document(tmp_path: Path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n   ")

    try:
        load_and_split(f, doc_id="doc-empty", filename="empty.txt")
        assert False, "expected ValueError for empty document"
    except ValueError:
        pass
