"""
Pytest configuration.

Sets up an isolated, network-free test environment *before* any `app.*`
module is imported (env vars are applied at module import time, which
happens before fixtures run), then provides fakes for the two external
dependencies — the embedding model and the LLM — so the test suite is
fast, deterministic, and runnable with zero internet access / API keys.
"""
import os
import tempfile
from pathlib import Path

_TMP_DIR = Path(tempfile.mkdtemp(prefix="rag_chatbot_test_"))
os.environ["GROQ_API_KEY"] = "test-key"
os.environ["UPLOAD_DIR"] = str(_TMP_DIR / "uploads")
os.environ["CHROMA_PERSIST_DIR"] = str(_TMP_DIR / "chroma_db")
os.environ["SQLITE_DB_PATH"] = str(_TMP_DIR / "app.db")

import pytest  # noqa: E402  (must come after the env vars are set above)


class FakeEmbeddings:
    """Deterministic, dependency-free stand-in for HuggingFaceEmbeddings.

    Lets ingestion/vector-store tests exercise real Chroma read/write
    paths without downloading a sentence-transformers model or needing
    network access in CI.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        h = abs(hash(text))
        return [((h >> (i * 4)) % 97) / 97.0 for i in range(16)]


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    from app import vectorstore

    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: FakeEmbeddings())
    vectorstore.get_vectorstore.cache_clear()
    yield
    vectorstore.get_vectorstore.cache_clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app import db
    from app.main import app

    db.init_db()
    with TestClient(app) as test_client:
        yield test_client
