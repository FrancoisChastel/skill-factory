"""LLM client abstraction. Backends are injected so the core stays testable."""

from skill_factory.llm.client import AnthropicClient, FakeLLMClient, LLMClient
from skill_factory.llm.openai_compat import OpenAICompatibleClient
from skill_factory.llm.factory import SUPPORTED_PROVIDERS, make_client

__all__ = [
    "LLMClient",
    "FakeLLMClient",
    "AnthropicClient",
    "OpenAICompatibleClient",
    "make_client",
    "SUPPORTED_PROVIDERS",
]
