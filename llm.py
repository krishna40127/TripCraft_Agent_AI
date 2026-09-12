"""
Provider-agnostic LLM client factory.

TripCraft is built to run against either:
  - Groq (cloud, fast, generous free tier) -- the default, or
  - Ollama (fully local, zero API cost)

Every agent asks for a chat model through `get_llm()` instead of importing a
provider SDK directly, so swapping providers is a single environment variable
(`LLM_PROVIDER=groq` | `LLM_PROVIDER=ollama`) and nothing in backend.py changes.
"""
from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr


class LLMConfigError(RuntimeError):
    """Raised when the configured LLM provider is missing required config."""


def _build_groq(temperature: float, fast: bool) -> BaseChatModel:
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise LLMConfigError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com/keys and put it in .env, "
            "or set LLM_PROVIDER=ollama to run fully local."
        )
    model = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b") if fast else os.getenv(
        "GROQ_MODEL", "openai/gpt-oss-120b"
    )
    return ChatGroq(model=model, api_key=SecretStr(api_key), temperature=temperature)


def _build_ollama(temperature: float, fast: bool) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


@lru_cache(maxsize=8)
def _cached_llm(provider: str, temperature: float, fast: bool) -> BaseChatModel:
    if provider == "groq":
        return _build_groq(temperature, fast)
    if provider == "ollama":
        return _build_ollama(temperature, fast)
    raise LLMConfigError(f"Unknown LLM_PROVIDER='{provider}'. Use 'groq' or 'ollama'.")


def get_llm(temperature: float = 0.3, fast: bool = False) -> BaseChatModel:
    """Return a LangChain chat model for the configured provider.

    Args:
        temperature: sampling temperature.
        fast: use the cheaper/smaller model variant (guardrail checks, routing)
              instead of the main reasoning model (drafting agents).
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    return _cached_llm(provider, temperature, fast)


def provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "groq").lower().strip()


def extract_text(content: object) -> str:
    """A LangChain message's `.content` is typed as `str | list[str | dict]`
    -- most providers return a plain string, but multi-modal-capable ones can
    return a list of content blocks instead. Normalize either shape to a
    single string so callers can safely call `.strip()`/`.upper()` etc.
    without a type-checker complaint or a runtime AttributeError."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content) if content is not None else ""
