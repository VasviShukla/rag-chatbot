"""
Centralized application configuration.

All settings are loaded from environment variables (or a local .env file)
via pydantic-settings. Keeping configuration in one typed object — instead
of scattering os.environ[...] calls through the codebase — makes the app
easier to test (override Settings in fixtures) and easier to deploy
(swap a .env file, no code changes).
"""
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Groq free tier) ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.2

    # --- Embeddings (local, free, no API key needed) ---
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage paths ---
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "data" / "chroma_db"
    SQLITE_DB_PATH: Path = BASE_DIR / "data" / "app.db"
    CHROMA_COLLECTION_NAME: str = "documents"

    # --- Chunking / retrieval ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    RETRIEVER_TOP_K: int = 4
    CHAT_HISTORY_WINDOW: int = 10  # number of past turns sent to the LLM for context

    # --- Upload limits ---
    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".md", ".docx")

    # --- API / CORS ---
    CORS_ORIGINS: tuple[str, ...] = ("*",)
    API_TITLE: str = "RAG Chatbot API"
    API_VERSION: str = "1.0.0"

    def ensure_directories(self) -> None:
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Settings are cached as a singleton; tests can call get_settings.cache_clear()."""
    settings = Settings()
    settings.ensure_directories()
    return settings
