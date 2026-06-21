"""
Streamlit chat UI for the RAG chatbot.

Talks to the FastAPI backend over HTTP only — no LangChain/LLM logic lives
here, which keeps the UI swappable (could be replaced with React, a CLI,
a Slack bot, etc. without touching the backend at all).

Run with:  streamlit run frontend/streamlit_app.py
"""
import json
import os
import uuid

import requests
import streamlit as st


def _get_backend_url() -> str:
    # Streamlit Community Cloud injects values from its "Secrets" manager
    # into st.secrets, not into the process environment, so check both.
    try:
        if "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"]
    except Exception:
        pass
    return os.environ.get("BACKEND_URL", "http://localhost:8000")


BACKEND_URL = _get_backend_url()

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str, "sources": [...]}]


def get_documents():
    try:
        resp = requests.get(f"{BACKEND_URL}/api/documents", timeout=10)
        resp.raise_for_status()
        return resp.json()["documents"]
    except requests.RequestException as e:
        st.sidebar.error(f"Could not reach backend: {e}")
        return []


# --------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("📄 Documents")

    uploaded_file = st.file_uploader(
        "Upload a document", type=["pdf", "txt", "md", "docx"]
    )
    if uploaded_file is not None:
        if st.button("Upload", use_container_width=True):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            with st.spinner("Uploading..."):
                resp = requests.post(f"{BACKEND_URL}/api/documents/upload", files=files)
            if resp.ok:
                st.success(f"Uploaded {uploaded_file.name}")
            else:
                st.error(resp.json().get("detail", resp.text))

    st.divider()
    documents = get_documents()

    pending = [d for d in documents if d["status"] in ("uploaded", "failed")]
    if pending:
        if st.button(f"⚙️ Ingest {len(pending)} pending document(s)", use_container_width=True):
            with st.spinner("Chunking + embedding..."):
                resp = requests.post(f"{BACKEND_URL}/api/documents/ingest", json={})
            if resp.ok:
                for r in resp.json()["results"]:
                    if r["status"] == "ingested":
                        st.success(f"{r['filename']}: {r['chunk_count']} chunks")
                    else:
                        st.error(f"{r['filename']}: {r['error']}")
            else:
                st.error(resp.text)

    st.caption("Your library")
    for doc in documents:
        icon = {"ingested": "✅", "uploaded": "⏳", "failed": "❌", "ingesting": "🔄"}.get(
            doc["status"], "•"
        )
        col1, col2 = st.columns([5, 1])
        with col1:
            st.write(f"{icon} {doc['filename']}")
            if doc["status"] == "ingested":
                st.caption(f"{doc['chunk_count']} chunks")
            elif doc["error"]:
                st.caption(doc["error"])
        with col2:
            if st.button("🗑", key=f"del_{doc['doc_id']}"):
                requests.delete(f"{BACKEND_URL}/api/documents/{doc['doc_id']}")
                st.rerun()

    st.divider()
    if st.button("🧹 Clear conversation", use_container_width=True):
        requests.delete(f"{BACKEND_URL}/api/chat/history/{st.session_state.session_id}")
        st.session_state.messages = []
        st.rerun()


# ------------------------------------------------------------------ main

st.title("💬 Chat with your documents")
st.caption("RAG chatbot — LangChain + FastAPI + Chroma + Groq (free tier)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s)"):
                for s in msg["sources"]:
                    page_info = f", page {s['page']}" if s.get("page") is not None else ""
                    st.markdown(f"**{s['filename']}** (chunk {s['chunk_index']}{page_info})")
                    st.caption(s["snippet"])

prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        sources = []
        try:
            with requests.post(
                f"{BACKEND_URL}/api/chat/stream",
                json={"session_id": st.session_state.session_id, "message": prompt},
                stream=True,
                timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    full_answer = f"⚠️ Backend error: {resp.text}"
                    placeholder.markdown(full_answer)
                else:
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[len("data: "):]
                        if data == "[DONE]":
                            break
                        event = json.loads(data)
                        if event["type"] == "token":
                            full_answer += event["content"]
                            placeholder.markdown(full_answer + "▌")
                        elif event["type"] == "sources":
                            sources = event["sources"]
                        elif event["type"] == "error":
                            full_answer += f"\n\n⚠️ {event['content']}"
                    placeholder.markdown(full_answer)
        except requests.RequestException as e:
            full_answer = f"⚠️ Could not reach backend: {e}"
            placeholder.markdown(full_answer)

        if sources:
            with st.expander(f"📚 {len(sources)} source(s)"):
                for s in sources:
                    page_info = f", page {s['page']}" if s.get("page") is not None else ""
                    st.markdown(f"**{s['filename']}** (chunk {s['chunk_index']}{page_info})")
                    st.caption(s["snippet"])

    st.session_state.messages.append(
        {"role": "assistant", "content": full_answer, "sources": sources}
    )
