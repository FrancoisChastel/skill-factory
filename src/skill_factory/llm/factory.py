"""Provider factory: build an LLMClient from a provider name + config.

Keeps provider selection in one place so configs and the CLI/UI can name a
provider by string (``anthropic``, ``openai``, ``ollama``, ...) without importing
backend classes.
"""

from __future__ import annotations

import os

from skill_factory.llm.client import AnthropicClient, LLMClient
from skill_factory.llm.openai_compat import OpenAICompatibleClient

# Providers that speak the OpenAI Chat Completions API, with sensible default
# base URLs (None => use the SDK/library default or the OPENAI_BASE_URL env var).
_OPENAI_COMPATIBLE: dict[str, str | None] = {
    "openai": None,
    "openai-compatible": None,
    "azure": None,  # set base_url explicitly to your Azure endpoint
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434/v1",
}

SUPPORTED_PROVIDERS = ("anthropic", *_OPENAI_COMPATIBLE.keys())


def make_client(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: object,
) -> LLMClient:
    """Construct an :class:`LLMClient` for ``provider``.

    Args:
        provider: One of :data:`SUPPORTED_PROVIDERS`.
        model: Model id for that provider.
        api_key: Overrides the provider's default env var.
        base_url: Overrides the provider's default endpoint (required for Azure
            and custom OpenAI-compatible servers).

    Raises:
        ValueError: for an unknown provider.
    """
    key = provider.strip().lower()
    if key == "anthropic":
        return AnthropicClient(
            model=model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            **_filtered(kwargs, {"max_retries", "retry_base_delay"}),
        )
    if key in _OPENAI_COMPATIBLE:
        resolved_base = base_url or _OPENAI_COMPATIBLE[key] or os.getenv("OPENAI_BASE_URL")
        return OpenAICompatibleClient(
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=resolved_base,
            **_filtered(kwargs, {"max_retries", "retry_base_delay", "extra_body"}),
        )
    raise ValueError(
        f"Unknown provider {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )


def _filtered(kwargs: dict[str, object], allowed: set[str]) -> dict[str, object]:
    return {k: v for k, v in kwargs.items() if k in allowed}
