"""Settings loaded from environment variables (and a local .env file, if present)."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
CATEGORIES_FILE = PROJECT_ROOT / "config" / "categories.yaml"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_base_url: str
    llm_model: str
    judge_model: str
    embedding_model: str
    embedding_dim: int
    qdrant_url: str
    qdrant_api_key: str
    qdrant_local_path: str
    qdrant_collection: str
    typesafe_api_key: str
    jev_model: str
    classifier_backend: str
    filter_mode: str


@lru_cache
def get_settings() -> Settings:
    env = os.getenv
    return Settings(
        openrouter_api_key=env("OPENROUTER_API_KEY", ""),
        openrouter_base_url=env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        llm_model=env("LLM_MODEL", "anthropic/claude-haiku-5.5"),
        judge_model=env("JUDGE_MODEL", "openai/gpt-6-luna"),
        embedding_model=env("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        embedding_dim=int(env("EMBEDDING_DIM", "1536")),
        qdrant_url=env("QDRANT_URL", ""),
        qdrant_api_key=env("QDRANT_API_KEY", ""),
        qdrant_local_path=str(PROJECT_ROOT / env("QDRANT_LOCAL_PATH", ".qdrant")),
        qdrant_collection=env("QDRANT_COLLECTION", "sungrid_kb"),
        typesafe_api_key=env("TYPESAFE_API_KEY", ""),
        jev_model=env("JEV_MODEL", "jev-latest"),
        classifier_backend=env("CLASSIFIER_BACKEND", "auto"),
        filter_mode=env("FILTER_MODE", "widened"),
    )


@lru_cache
def load_category_config() -> dict:
    """Parsed config/categories.yaml: {"categories": {...}, "documents": {...}}."""
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)
