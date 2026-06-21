"""
The actual RAG pipeline.

Built with LangChain's modern LCEL chain helpers:

  1. `create_history_aware_retriever` rewrites the latest user message into
     a standalone query using the conversation history (so "what about
     its pricing?" gets resolved against the previous turn before we
     embed and search).
  2. `create_stuff_documents_chain` stuffs the retrieved chunks into the
     prompt and asks the LLM to answer ONLY from that context.
  3. `create_retrieval_chain` wires the two together and also returns the
     retrieved `Document` objects alongside the answer, which is what
     lets us cite sources back to the user.

The LLM is Groq's hosted inference (free tier, OpenAI-compatible, very
low latency) via `langchain-groq`. Swapping providers later means
changing `get_llm()` only.
"""
from functools import lru_cache
from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from app import db
from app.config import get_settings
from app.logger import get_logger
from app.vectorstore import get_retriever

logger = get_logger(__name__)

CONTEXTUALIZE_PROMPT = (
    "Given a chat history and the latest user question, which might "
    "reference context in the chat history, rephrase the question into a "
    "standalone question that can be understood without the chat history. "
    "Do NOT answer the question — only rephrase it if needed, otherwise "
    "return it unchanged."
)

QA_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's own "
    "uploaded documents. Use ONLY the following retrieved context to answer. "
    "If the answer is not contained in the context, say you don't have "
    "enough information in the provided documents — do not make things up. "
    "Be concise and cite specific details from the context where relevant.\n\n"
    "Context:\n{context}"
)


@lru_cache
def get_llm(streaming: bool = False) -> ChatGroq:
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and put it in your .env file."
        )
    return ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        streaming=streaming,
    )


def _build_chain(streaming: bool = False):
    llm = get_llm(streaming=streaming)
    retriever = get_retriever()

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, answer_chain)


def _load_history(session_id: str) -> list:
    settings = get_settings()
    rows = db.get_chat_history(session_id, limit=settings.CHAT_HISTORY_WINDOW * 2)
    messages = []
    for row in rows:
        if row["role"] == "user":
            messages.append(HumanMessage(content=row["content"]))
        else:
            messages.append(AIMessage(content=row["content"]))
    return messages


def _sources_from_docs(docs) -> list[dict]:
    sources = []
    for d in docs:
        snippet = d.page_content[:300].strip()
        sources.append(
            {
                "doc_id": d.metadata.get("doc_id", "unknown"),
                "filename": d.metadata.get("filename", "unknown"),
                "chunk_index": d.metadata.get("chunk_index", -1),
                "page": d.metadata.get("page"),
                "snippet": snippet + ("..." if len(d.page_content) > 300 else ""),
            }
        )
    return sources


def get_chat_response(session_id: str, message: str) -> dict:
    """Non-streaming RAG call. Returns {"answer": str, "sources": [...]}"""
    chain = _build_chain(streaming=False)
    history = _load_history(session_id)

    result = chain.invoke({"input": message, "chat_history": history})

    db.add_chat_message(session_id, "user", message)
    db.add_chat_message(session_id, "assistant", result["answer"])

    return {
        "answer": result["answer"],
        "sources": _sources_from_docs(result.get("context", [])),
    }


def stream_chat_response(session_id: str, message: str) -> Iterator[dict]:
    """
    Streaming RAG call. Yields dicts of the shape:
      {"type": "token", "content": "..."}      -- incremental answer tokens
      {"type": "sources", "sources": [...]}     -- once, after generation completes
    The full answer is persisted to chat history once streaming finishes.
    """
    chain = _build_chain(streaming=True)
    history = _load_history(session_id)

    db.add_chat_message(session_id, "user", message)

    full_answer = []
    sources: list[dict] = []

    for chunk in chain.stream({"input": message, "chat_history": history}):
        if "context" in chunk and chunk["context"]:
            sources = _sources_from_docs(chunk["context"])
        if "answer" in chunk and chunk["answer"]:
            full_answer.append(chunk["answer"])
            yield {"type": "token", "content": chunk["answer"]}

    answer_text = "".join(full_answer)
    db.add_chat_message(session_id, "assistant", answer_text)

    yield {"type": "sources", "sources": sources}
