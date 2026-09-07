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
    LLM_PROVIDER: str = "groq"  # "groq" or "openai"
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    VISION_MODEL: str = "llama-3.2-11b-vision-preview"
    FAST_CHECK_MODEL: str = "llama-3.1-8b-instant"

    # Embedding & Vector Store
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    QDRANT_URL: Optional[str] = None  # If None, uses local QDRANT_STORAGE_PATH
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "fact_embeddings"
    SIMILARITY_THRESHOLD: float = 0.72

    # Database
    DATABASE_URL: str = "sqlite:///./data/fact_layer.db"

    # Execution limits
    MAX_EXTRACTION_WORKERS: int = 2
    IMAGE_DENSITY_THRESHOLD: float = 0.15  # Char count / (w * h / 100) below which page is image-heavy


settings = Settings()

# Ensure critical data directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.QDRANT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
