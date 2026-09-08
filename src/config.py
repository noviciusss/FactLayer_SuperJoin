"""Configuration settings for Fact Knowledge Layer."""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    PREVIEW_DIR: Path = DATA_DIR / "previews"
    CACHE_DIR: Path = DATA_DIR / "cache"
    QDRANT_STORAGE_PATH: Path = DATA_DIR / "qdrant"

    # LLM Settings
    LLM_PROVIDER: str = "groq"  # "groq", "azure", or "openai"
    AZURE_OPENAI_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-5"
    AZURE_DOC_INTELLIGENCE_ENDPOINT: Optional[str] = None
    AZURE_DOC_INTELLIGENCE_KEY: Optional[str] = None

    GROQ_API_KEY: Optional[str] = None
    GROQ_API_KEY_SECONDARY: Optional[str] = None
    GROQ_API_KEY_3: Optional[str] = None
    GROQ_API_KEY_4: Optional[str] = None
    GROQ_API_KEYS: Optional[str] = None  # Comma-separated list of keys
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "openai/gpt-oss-20b"
    VISION_MODEL: str = "qwen/qwen3.6-27b"
    ADJUDICATION_MODEL: str = "openai/gpt-oss-20b"
    FAST_CHECK_MODEL: str = "openai/gpt-oss-20b"

    # Embedding & Vector Store
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    QDRANT_URL: Optional[str] = None  # If None, uses local QDRANT_STORAGE_PATH
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "fact_embeddings"
    SIMILARITY_THRESHOLD: float = 0.72

    # Database
    DATABASE_URL: str = "sqlite:///./data/fact_layer.db"

    # Execution limits & timeouts
    MAX_EXTRACTION_WORKERS: int = 2
    IMAGE_DENSITY_THRESHOLD: float = 0.15  # Char count / (w * h / 100) below which page is image-heavy
    LLM_REQUEST_TIMEOUT: float = 60.0  # Explicit Groq / Azure / OpenAI HTTP timeout in seconds
    LLM_MAX_RETRIES: int = 3  # Max retries per call
    LLM_MAX_BACKOFF: float = 20.0  # Max single retry sleep ceiling
    DOCUMENT_PROCESSING_TIMEOUT: float = 300.0  # 5-minute backstop timeout per document
    MAX_CHUNK_SIZE: int = 2400  # Max characters per coalesced chunk to stay well under LLM context


settings = Settings()

# Ensure critical data directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.QDRANT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
