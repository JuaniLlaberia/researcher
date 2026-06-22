"""
Application configuration — single source of truth for environment settings.

Values are read from the process environment and the project `.env` file.
`load_dotenv()` is called so that existing `os.getenv` lookups (e.g. the provider
key checks in `llm.py` / `embedder.py`) continue to resolve from `.env` as well.
Shell-exported variables take precedence over `.env` (override=False).
"""
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=False)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- PostgreSQL ---
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "researcher"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # --- LLM ---
    llm_provider: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.0

    # --- Provider API keys (only the selected provider's key is required) ---
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # --- External paper sources (optional; raise rate limits when set) ---
    semantic_scholar_api_key: str = ""

    # --- Embedder ---
    embedder_provider: str = ""
    embedder_model: str = ""
    # Output dimension of the chosen embedder. MUST match the Vector(N) column on PaperChunk
    embedding_dim: int = 768

    # --- Reranker (cross-encoder, HuggingFace) ---
    reranker_model: str = "BAAI/bge-reranker-base"
    # Leave empty to auto-detect.
    reranker_device: str = ""

    # --- Logging ---
    log_level: str = "INFO"
    # If set, also write a plain (no-color) transcript of the run to this file.
    log_file: str = ""

    @computed_field
    @property
    def database_url_async(self) -> str:
        """
        asyncpg URL — the app runtime (graph nodes, FastAPI).
        """
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def database_url_sync(self) -> str:
        """psycopg URL — Alembic migrations and standalone scripts."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
