"""Pydantic settings for RAG system configuration."""

from functools import lru_cache
from typing import List, Literal

from pydantic_settings import BaseSettings


class RagSettings(BaseSettings):
    """Main RAG system configuration."""

    # LLM Provider Selection
    llm_provider: Literal["anthropic", "ollama", "openai"] = "anthropic"

    # Anthropic Configuration
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # LLM Parameters
    temperature: float = 0.7
    max_tokens: int = 2048

    # Vector Store Configuration
    vector_store_path: str = "data/vector_store"
    collection_name: str = "documents"
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    top_k_results: int = 5

    # Service Configuration
    port: int = 8000
    host: str = "0.0.0.0"
    session_dir: str = "data/sessions"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class IngestionSettings(BaseSettings):
    """Document ingestion configuration."""

    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: Literal["fixed", "semantic", "sentence"] = "fixed"

    # Batch Processing
    batch_size: int = 100
    max_workers: int = 4

    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32

    # File Support
    supported_formats: List[str] = [".pdf", ".docx", ".txt", ".md", ".html"]
    max_file_size_mb: int = 50

    # Processing Options
    remove_duplicates: bool = True
    normalize_text: bool = True

    class Config:
        env_file = ".env"
        env_prefix = "INGESTION_"
        extra = "ignore"


@lru_cache()
def get_settings() -> RagSettings:
    """Get cached RAG settings instance."""
    return RagSettings()


@lru_cache()
def get_ingestion_settings() -> IngestionSettings:
    """Get cached ingestion settings instance."""
    return IngestionSettings()
