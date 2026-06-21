"""
Lightweight SQLite persistence layer.

Why SQLite instead of an external database: this is a single-node RAG
service, so a zero-infrastructure embedded DB is the right tool for the
job. It gives us real SQL, transactions, and durability across restarts
without requiring the user to stand up (and pay for) Postgres/MySQL. The
access layer below is the only place that touches SQL, so swapping in a
"real" database later (e.g. for a multi-instance deployment) only means
rewriting this one module.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from app.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    path         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'uploaded',
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    uploaded_at  TEXT NOT NULL,
    ingested_at  TEXT
);

CREATE TABLE IF NOT EXISTS chat_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_history_session
    ON chat_history (session_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Optional[Path] = None) -> None:
    path = db_path or get_settings().SQLITE_DB_PATH
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


@contextmanager
def get_conn(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    path = db_path or get_settings().SQLITE_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------- documents

def create_document(doc_id: str, filename: str, path: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, filename, path, status, uploaded_at) "
            "VALUES (?, ?, ?, 'uploaded', ?)",
            (doc_id, filename, path, _now()),
        )
        conn.commit()


def set_document_status(
    doc_id: str,
    status: str,
    chunk_count: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    with get_conn() as conn:
        if status == "ingested":
            conn.execute(
                "UPDATE documents SET status = ?, chunk_count = ?, error = NULL, "
                "ingested_at = ? WHERE doc_id = ?",
                (status, chunk_count or 0, _now(), doc_id),
            )
        else:
            conn.execute(
                "UPDATE documents SET status = ?, error = ? WHERE doc_id = ?",
                (status, error, doc_id),
            )
        conn.commit()


def get_document(doc_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row


def list_documents(statuses: Optional[list[str]] = None) -> list[sqlite3.Row]:
    with get_conn() as conn:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            rows = conn.execute(
                f"SELECT * FROM documents WHERE status IN ({placeholders}) "
                "ORDER BY uploaded_at DESC",
                statuses,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
        return rows


def delete_document(doc_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()


def aggregate_stats() -> tuple[int, int]:
    """Returns (documents_ingested, total_chunks)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS docs, COALESCE(SUM(chunk_count), 0) AS chunks "
            "FROM documents WHERE status = 'ingested'"
        ).fetchone()
        return row["docs"], row["chunks"]


# ------------------------------------------------------------- chat history

def add_chat_message(session_id: str, role: str, content: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )
        conn.commit()


def get_chat_history(session_id: str, limit: Optional[int] = None) -> list[sqlite3.Row]:
    with get_conn() as conn:
        if limit:
            rows = conn.execute(
                "SELECT * FROM chat_history WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(
                "SELECT * FROM chat_history WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return rows


def clear_chat_history(session_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.commit()
