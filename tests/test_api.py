def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "llm_model" in body


def test_upload_rejects_unsupported_extension(client):
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("note.xyz", b"hello", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_upload_list_ingest_delete_document(client):
    content = (
        "RAG stands for Retrieval-Augmented Generation. " * 5
        + "\n\n"
        + "It combines a retriever with a generator LLM. " * 5
    ).encode("utf-8")

    upload_resp = client.post(
        "/api/documents/upload",
        files={"file": ("rag_notes.txt", content, "text/plain")},
    )
    assert upload_resp.status_code == 200
    doc = upload_resp.json()
    assert doc["status"] == "uploaded"
    doc_id = doc["doc_id"]

    list_resp = client.get("/api/documents")
    assert list_resp.status_code == 200
    assert any(d["doc_id"] == doc_id for d in list_resp.json()["documents"])

    ingest_resp = client.post("/api/documents/ingest", json={"doc_ids": [doc_id]})
    assert ingest_resp.status_code == 200
    results = ingest_resp.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "ingested"
    assert results[0]["chunk_count"] > 0

    delete_resp = client.delete(f"/api/documents/{doc_id}")
    assert delete_resp.status_code == 200

    list_resp_after = client.get("/api/documents")
    assert all(d["doc_id"] != doc_id for d in list_resp_after.json()["documents"])


def test_delete_unknown_document_returns_404(client):
    resp = client.delete("/api/documents/does-not-exist")
    assert resp.status_code == 404


def test_chat_requires_ingested_documents(client, monkeypatch):
    from app import main as main_module

    monkeypatch.setattr(main_module, "collection_chunk_count", lambda: 0)

    resp = client.post("/api/chat", json={"session_id": "s1", "message": "hi"})
    assert resp.status_code == 400


def test_chat_returns_answer_and_sources_with_mocked_llm(client, monkeypatch):
    """
    The Groq LLM call itself is mocked out here: this test verifies the
    HTTP contract (request/response shape, status codes, history
    persistence) rather than real model output, which keeps the suite
    fast, free, and deterministic in CI.
    """
    from app import main as main_module

    monkeypatch.setattr(main_module, "collection_chunk_count", lambda: 1)
    monkeypatch.setattr(
        main_module,
        "get_chat_response",
        lambda session_id, message: {
            "answer": "RAG combines retrieval with generation.",
            "sources": [
                {
                    "doc_id": "doc-1",
                    "filename": "rag_notes.txt",
                    "chunk_index": 0,
                    "page": None,
                    "snippet": "RAG stands for Retrieval-Augmented Generation...",
                }
            ],
        },
    )

    resp = client.post(
        "/api/chat", json={"session_id": "session-A", "message": "What is RAG?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "session-A"
    assert "Retrieval-Augmented" in body["answer"] or "retrieval" in body["answer"].lower()
    assert body["sources"][0]["filename"] == "rag_notes.txt"


def test_chat_history_lifecycle(client):
    from app import db

    session_id = "session-history-test"
    db.add_chat_message(session_id, "user", "hello")
    db.add_chat_message(session_id, "assistant", "hi there")

    resp = client.get(f"/api/chat/history/{session_id}")
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]

    clear_resp = client.delete(f"/api/chat/history/{session_id}")
    assert clear_resp.status_code == 200

    resp_after = client.get(f"/api/chat/history/{session_id}")
    assert resp_after.json()["messages"] == []
