"""Text embeddings via OpenRouter's OpenAI-compatible /embeddings endpoint."""

from functools import lru_cache

from openai import OpenAI

from sungrid_copilot.config import get_settings

BATCH_SIZE = 64


@lru_cache
def _client() -> OpenAI:
    s = get_settings()
    if not s.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — see .env.example")
    return OpenAI(api_key=s.openrouter_api_key, base_url=s.openrouter_base_url)


def embed_texts(texts: list[str]) -> list[list[float]]:
    s = get_settings()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = _client().embeddings.create(model=s.embedding_model, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    for v in vectors:
        if len(v) != s.embedding_dim:
            raise ValueError(f"Embedding dim {len(v)} != EMBEDDING_DIM {s.embedding_dim}")
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
