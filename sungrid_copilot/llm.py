"""Chat models via OpenRouter (OpenAI-compatible), wrapped as LangChain chat models."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from sungrid_copilot.config import get_settings


@lru_cache  # reuse clients (and their HTTP connections) instead of a new TLS handshake per call
def get_chat_model(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    s = get_settings()
    if not s.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — see .env.example")
    return ChatOpenAI(
        model=model or s.llm_model,
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )
